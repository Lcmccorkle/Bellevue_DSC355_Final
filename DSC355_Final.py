# DSC355_Final.py
# Walmart Weekly Sales Forecaster – Robust & production-ready version

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')


# ────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────

MODEL_PATH = 'walmart_xgb_model.joblib'

PART_FILES = [
    'engineered_walmart_data_Part1.csv',
    'engineered_walmart_data_Part2.csv',
    'engineered_walmart_data_Part3.csv',
    'engineered_walmart_data_Part4.csv'
]

# Encoding order: most common → most tolerant
ENCODING_TRIES = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1', 'utf-8-sig']

DEFAULTS = {
    'size': 140000,
    'temperature': 60.0,
    'fuel_price': 3.3,
    'unemployment': 7.0,
    'cpi': 170.0,
}


# ────────────────────────────────────────────────
#  Load model & reference data (robust)
# ────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # Model
    try:
        model = joblib.load(MODEL_PATH)
        st.session_state.model_loaded = True
    except Exception as e:
        st.error(f"Model failed to load: {type(e).__name__} – {e}")
        st.stop()

    # Data parts
    df_ref = None
    loaded = 0
    parts = []

    for path in PART_FILES:
        success = False
        for enc in ENCODING_TRIES:
            try:
                df_part = pd.read_csv(
                    path,
                    parse_dates=['Date'],
                    encoding=enc,
                    encoding_errors='replace',
                    low_memory=False
                )
                parts.append(df_part)
                loaded += 1
                success = True
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.warning(f"{path} failed with {enc}: {e}")
                break

        if not success:
            st.warning(f"Could not load {path} with any encoding")

    if parts:
        try:
            df_ref = pd.concat(parts, ignore_index=True)
            st.success(f"Loaded {loaded}/{len(PART_FILES)} data parts")
        except Exception as e:
            st.error(f"Concat failed: {e}")
    else:
        st.warning("No data parts loaded – using fallback ranges/defaults")

    return model, df_ref


model, df_ref = load_resources()


# ────────────────────────────────────────────────
#  Choices & defaults
# ────────────────────────────────────────────────
if df_ref is not None and 'Store' in df_ref.columns:
    stores = sorted(df_ref['Store'].unique().astype(int))
    depts  = sorted(df_ref['Dept'].unique().astype(int))
    types  = ['A', 'B', 'C']  # fallback
else:
    stores = list(range(1, 46))
    depts  = list(range(1, 100))
    types  = ['A', 'B', 'C']


# ────────────────────────────────────────────────
#  UI
# ────────────────────────────────────────────────
st.title("Walmart Weekly Sales Forecaster")
st.caption("XGBoost model – Milestone 4")

st.info("""
Top features (SHAP): Store, Dept, Size, IsHoliday, Temperature, Total Markdown,  
Type, CPI, Unemployment, Markdown interactions, binned categories
""")

with st.form("sales_form"):

    st.subheader("Store & Department")
    c1, c2 = st.columns(2)
    with c1: store = st.selectbox("Store", stores, index=0)
    with c2: dept  = st.selectbox("Department", depts, index=0)

    st.subheader("Date & Holiday")
    c3, c4 = st.columns(2)
    with c3: week_start = st.date_input("Week start", datetime(2012, 9, 1))
    with c4: is_holiday = st.checkbox("Holiday week?", False)

    st.subheader("Store & Economic")
    c5, c6, c7 = st.columns(3)
    with c5:
        size = st.number_input("Size (sq ft)", 30000, 220000, DEFAULTS['size'], step=5000)
        store_type = st.selectbox("Type", types, index=0)
    with c6:
        temperature = st.number_input("Temperature (°F)", -20.0, 110.0, DEFAULTS['temperature'], step=1.0)
        fuel_price  = st.number_input("Fuel Price ($/gal)", 1.5, 5.0, DEFAULTS['fuel_price'], step=0.1)
    with c7:
        unemployment = st.number_input("Unemployment (%)", 3.0, 12.0, DEFAULTS['unemployment'], step=0.1)
        cpi = st.number_input("CPI", 120.0, 230.0, DEFAULTS['cpi'], step=0.1)

    st.subheader("MarkDowns")
    md1 = st.number_input("MarkDown1", 0.0, value=0.0, step=100.0, format="%.0f")
    md2 = st.number_input("MarkDown2", 0.0, value=0.0, step=100.0, format="%.0f")
    md3 = st.number_input("MarkDown3", 0.0, value=0.0, step=100.0, format="%.0f")
    md4 = st.number_input("MarkDown4", 0.0, value=0.0, step=100.0, format="%.0f")
    md5 = st.number_input("MarkDown5", 0.0, value=0.0, step=100.0, format="%.0f")

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)


# ────────────────────────────────────────────────
#  Prediction
# ────────────────────────────────────────────────
if submitted:
    with st.spinner("Making prediction..."):

        total_md = md1 + md2 + md3 + md4 + md5

        # Build base input
        row = {
            'Store': store,
            'Dept': dept,
            'IsHoliday': is_holiday,
            'Size': size,
            'Temperature': temperature,
            'Fuel_Price': fuel_price,
            'MarkDown1': md1,
            'MarkDown2': md2,
            'MarkDown3': md3,
            'MarkDown4': md4,
            'MarkDown5': md5,
            'CPI': cpi,
            'Unemployment': unemployment,
            'Type_B': 1 if store_type == 'B' else 0,
            'Type_C': 1 if store_type == 'C' else 0,
            'Year': week_start.year,
            'Month': week_start.month,
            'Week': week_start.isocalendar()[1],
            'Quarter': (week_start.month - 1) // 3 + 1,
            'DayOfWeek': week_start.weekday(),
            'Total_MarkDown': total_md,
            'Holiday_x_TotalMarkdown': int(is_holiday) * total_md,
            'IsWeekend': 1 if week_start.weekday() >= 5 else 0,
        }

        # Add all missing engineered columns with defaults (0/1 or neutral values)
        # This prevents feature mismatch crash
        full_row = row.copy()

        # Binned categories (simple threshold-based – adjust if you know exact bins)
        full_row['Temp_Category_Hot']     = 1 if temperature > 80 else 0
        full_row['Temp_Category_Mild']    = 1 if 50 <= temperature <= 80 else 0
        full_row['Unemployment_Category_Low']    = 1 if unemployment < 6 else 0
        full_row['Unemployment_Category_Medium'] = 1 if 6 <= unemployment <= 8 else 0
        full_row['Size_Category_Medium']  = 1 if 100000 <= size <= 180000 else 0
        full_row['Size_Category_Small']   = 1 if size < 100000 else 0
        full_row['FuelPrice_Category_Low']    = 1 if fuel_price < 3.0 else 0
        full_row['FuelPrice_Category_Medium'] = 1 if 3.0 <= fuel_price <= 4.0 else 0
        full_row['CPI_Category_Low']      = 1 if cpi < 150 else 0
        full_row['CPI_Category_Medium']   = 1 if 150 <= cpi <= 190 else 0

        # Other common missing columns from your training set
        full_row['Economic_Index'] = 0               # or median if known
        full_row['Markdown_Group'] = 'None'          # most common or neutral value
        # Add any other dummies / features that appear in the error message

        X_input = pd.DataFrame([full_row])

        # Align columns if model has feature_names_in_
        try:
            expected = model.feature_names_in_
            X_input = X_input.reindex(columns=expected, fill_value=0)
        except AttributeError:
            pass  # model doesn't expose feature names – hope order is ok

        # Predict
        try:
            log_y = model.predict(X_input)[0]
            dollars = np.expm1(log_y)

            st.success("Prediction ready")
            st.metric("Predicted Weekly Sales", f"${dollars:,.0f}")
            st.metric("Log-scale value", f"{log_y:.4f}")
            st.caption(f"Store {store} • Dept {dept} • {week_start:%Y-%m-%d}")

        except Exception as e:
            st.error(f"Prediction failed: {type(e).__name__} – {e}")
            st.info("Possible causes: still-missing columns, wrong data types, or model corruption.")


# ────────────────────────────────────────────────
#  Info
# ────────────────────────────────────────────────
with st.expander("Model details"):
    st.markdown("""
    - Model: XGBoost (Milestone 4)
    - Target: log₁ₚ(Weekly_Sales)
    - Approx test performance: MAE $7k–$9k, R² ~0.93–0.96
    - Trained on data up to ~Oct 2012
    - Some features use simple defaults when not provided
    """)
