# DSC355_Final.py
# Walmart Weekly Sales Forecaster – Final Robust Version

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings

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

# Encoding order: try common ones until one works
ENCODING_ORDER = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1', 'utf-8-sig']

# Default values for numeric / non-provided columns
FALLBACK_DEFAULTS = {
    'Store': 1,
    'Dept': 1,
    'IsHoliday': False,
    'Size': 140000,
    'Temperature': 60.0,
    'Fuel_Price': 3.3,
    'MarkDown1': 0.0,
    'MarkDown2': 0.0,
    'MarkDown3': 0.0,
    'MarkDown4': 0.0,
    'MarkDown5': 0.0,
    'CPI': 170.0,
    'Unemployment': 7.0,
    'Type_B': 0,
    'Type_C': 0,
    'Year': 2012,
    'Month': 9,
    'Week': 36,
    'Quarter': 3,
    'DayOfWeek': 4,
    'Total_MarkDown': 0.0,
    'Holiday_x_TotalMarkdown': 0.0,
    'IsWeekend': 0,
    'Economic_Index': 0.0,
    'Markdown_Group': 'None',  # most common / neutral value
}


# ────────────────────────────────────────────────
#  Load model & reference data
# ────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # Load model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {type(e).__name__} – {e}")
        st.stop()

    # Load data parts (robust encoding + fallback)
    df_ref = None
    parts_loaded = []

    for path in PART_FILES:
        success = False
        for enc in ENCODING_ORDER:
            try:
                df_part = pd.read_csv(
                    path,
                    parse_dates=['Date'],
                    encoding=enc,
                    encoding_errors='replace',
                    low_memory=False
                )
                parts_loaded.append(df_part)
                success = True
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.warning(f"{path} failed with {enc}: {e}")
                break

        if not success:
            st.warning(f"Skipped {path} – could not load with any encoding")

    if parts_loaded:
        try:
            df_ref = pd.concat(parts_loaded, ignore_index=True)
            st.success(f"Loaded {len(parts_loaded)} / {len(PART_FILES)} data parts")
        except Exception as e:
            st.error(f"Concatenation failed: {e}")
    else:
        st.warning("No data parts loaded – dropdowns use fallback ranges")

    return model, df_ref


model, df_ref = load_resources()


# ────────────────────────────────────────────────
#  Dropdown choices
# ────────────────────────────────────────────────
if df_ref is not None and 'Store' in df_ref.columns:
    stores = sorted(df_ref['Store'].unique().astype(int))
    depts = sorted(df_ref['Dept'].unique().astype(int))
    types = ['A', 'B', 'C']
else:
    stores = list(range(1, 46))
    depts = list(range(1, 100))
    types = ['A', 'B', 'C']


# ────────────────────────────────────────────────
#  UI
# ────────────────────────────────────────────────
st.title("Walmart Weekly Sales Forecaster")
st.markdown("Predict department-level weekly sales (Milestone 4 XGBoost model)")

with st.form("sales_form"):

    st.subheader("Store & Department")
    c1, c2 = st.columns(2)
    with c1: store = st.selectbox("Store", stores, index=0)
    with c2: dept = st.selectbox("Department", depts, index=0)

    st.subheader("Date & Holiday")
    c3, c4 = st.columns(2)
    with c3: week_start = st.date_input("Week start", datetime(2012, 9, 1))
    with c4: is_holiday = st.checkbox("Holiday week?", False)

    st.subheader("Store & Economic Features")
    c5, c6, c7 = st.columns(3)
    with c5:
        size = st.number_input("Store Size (sq ft)", 30000, 220000, FALLBACK_DEFAULTS['size'], step=5000)
        store_type = st.selectbox("Store Type", types, index=0)
    with c6:
        temperature = st.number_input("Temperature (°F)", -20.0, 110.0, FALLBACK_DEFAULTS['temperature'], step=1.0)
        fuel_price = st.number_input("Fuel Price ($/gal)", 1.5, 5.0, FALLBACK_DEFAULTS['fuel_price'], step=0.1)
    with c7:
        unemployment = st.number_input("Unemployment (%)", 3.0, 12.0, FALLBACK_DEFAULTS['unemployment'], step=0.1)
        cpi = st.number_input("CPI", 120.0, 230.0, FALLBACK_DEFAULTS['cpi'], step=0.1)

    st.subheader("MarkDowns")
    md1 = st.number_input("MarkDown1 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md2 = st.number_input("MarkDown2 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md3 = st.number_input("MarkDown3 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md4 = st.number_input("MarkDown4 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md5 = st.number_input("MarkDown5 ($)", 0.0, value=0.0, step=100.0, format="%.0f")

    submitted = st.form_submit_button("Predict Weekly Sales", type="primary", use_container_width=True)


# ────────────────────────────────────────────────
#  Prediction
# ────────────────────────────────────────────────
if submitted:
    with st.spinner("Preparing features and predicting..."):

        total_md = md1 + md2 + md3 + md4 + md5

        # Base input row
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
            'Economic_Index': 0.0,
            'Markdown_Group': 'None',  # ← string value
        }

        X_input = pd.DataFrame([row])

        # ────────────────────────────────────────────────
        #  FIX CATEGORICAL DTYPE (critical for enable_categorical=True)
        # ────────────────────────────────────────────────
        categorical_cols = ['Markdown_Group']  # add more if you know them

        for col in categorical_cols:
            if col in X_input.columns:
                X_input[col] = X_input[col].astype('category')

        # Add all other expected binned/dummy columns with defaults
        X_input['Temp_Category_Hot'] = 1 if temperature > 80 else 0
        X_input['Temp_Category_Mild'] = 1 if 50 <= temperature <= 80 else 0
        X_input['Unemployment_Category_Low'] = 1 if unemployment < 6 else 0
        X_input['Unemployment_Category_Medium'] = 1 if 6 <= unemployment <= 8 else 0
        X_input['Size_Category_Medium'] = 1 if 100000 <= size <= 180000 else 0
        X_input['Size_Category_Small'] = 1 if size < 100000 else 0
        X_input['FuelPrice_Category_Low'] = 1 if fuel_price < 3.0 else 0
        X_input['FuelPrice_Category_Medium'] = 1 if 3.0 <= fuel_price <= 4.0 else 0
        X_input['CPI_Category_Low'] = 1 if cpi < 150 else 0
        X_input['CPI_Category_Medium'] = 1 if 150 <= cpi <= 190 else 0

        # Align to model's expected columns
        try:
            expected = model.feature_names_in_
            X_input = X_input.reindex(columns=expected, fill_value=0)

            # Re-apply category dtype after reindex (fill_value can reset it)
            for col in categorical_cols:
                if col in X_input.columns:
                    X_input[col] = X_input[col].astype('category')

        except AttributeError:
            st.info("Model does not expose feature_names_in_ → using current columns")

        # Make prediction
        try:
            log_y = model.predict(X_input)[0]
            dollars = np.expm1(log_y)

            st.success("Prediction successful")
            st.metric("Predicted Weekly Sales", f"${dollars:,.0f}")
            st.metric("Log prediction (internal)", f"{log_y:.4f}")
            st.caption(f"Store {store} • Dept {dept} • Week of {week_start:%Y-%m-%d}")

        except Exception as e:
            st.error(f"Prediction failed: {type(e).__name__} – {e}")
            st.info("If dtype-related: check categorical columns like Markdown_Group")


# ────────────────────────────────────────────────
#  Footer / Info
# ────────────────────────────────────────────────
with st.expander("Model details"):
    st.markdown("""
    - **Model**: XGBoost Regressor (Milestone 4)
    - **Target**: log₁ₚ(Weekly_Sales)
    - **Approx test performance**: MAE $7,000–$9,000, R² 0.93–0.96
    - **Training data**: up to ~Oct 2012
    - Some engineered features use defaults when not provided
    """)
