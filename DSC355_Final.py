import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

MODEL_PATH = 'walmart_xgb_model.joblib'

PART_FILES = [
    'engineered_walmart_data_Part1.csv',
    'engineered_walmart_data_Part2.csv',
    'engineered_walmart_data_Part3.csv',
    'engineered_walmart_data_Part4.csv'
]

ENCODING_TRIES = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1', 'utf-8-sig']

# Define fallback defaults BEFORE using them
FALLBACK_DEFAULTS = {
    'size': 140000,
    'temperature': 60.0,
    'fuel_price': 3.3,
    'unemployment': 7.0,
    'cpi': 170.0,
}


# ────────────────────────────────────────────────
# Load model & reference data
# ────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

    df_ref = None
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
                success = True
                break
            except:
                continue

        if not success:
            st.warning(f"Skipped {path}")

    if parts:
        df_ref = pd.concat(parts, ignore_index=True)

    return model, df_ref


model, df_ref = load_resources()


# ────────────────────────────────────────────────
# Dropdown choices
# ────────────────────────────────────────────────
if df_ref is not None and 'Store' in df_ref.columns:
    stores = sorted(df_ref['Store'].astype(int).unique())
    depts = sorted(df_ref['Dept'].astype(int).unique())
    types = ['A', 'B', 'C']
else:
    stores = list(range(1, 46))
    depts = list(range(1, 100))
    types = ['A', 'B', 'C']


# ────────────────────────────────────────────────
# UI + FORM
# ────────────────────────────────────────────────
st.title("Walmart Weekly Sales Forecaster")

with st.form("prediction_form"):

    st.subheader("Store & Department")
    col1, col2 = st.columns(2)
    with col1:
        store = st.selectbox("Store", stores, index=0)
    with col2:
        dept = st.selectbox("Department", depts, index=0)

    st.subheader("Date & Holiday")
    col3, col4 = st.columns(2)
    with col3:
        week_start = st.date_input("Week start date", datetime(2012, 9, 1))
    with col4:
        is_holiday = st.checkbox("Holiday week?", False)

    st.subheader("Store & Economic Features")
    col5, col6, col7 = st.columns(3)
    with col5:
        size = st.number_input("Store Size (sq ft)", 30000, 220000, FALLBACK_DEFAULTS['size'], step=5000)
        store_type = st.selectbox("Store Type", types, index=0)
    with col6:
        temperature = st.number_input("Temperature (°F)", -20.0, 110.0, FALLBACK_DEFAULTS['temperature'], step=1.0)
        fuel_price = st.number_input("Fuel Price ($/gal)", 1.5, 5.0, FALLBACK_DEFAULTS['fuel_price'], step=0.1)
    with col7:
        unemployment = st.number_input("Unemployment Rate (%)", 3.0, 12.0, FALLBACK_DEFAULTS['unemployment'], step=0.1)
        cpi = st.number_input("CPI", 120.0, 230.0, FALLBACK_DEFAULTS['cpi'], step=0.1)

    st.subheader("MarkDowns")
    md1 = st.number_input("MarkDown1 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md2 = st.number_input("MarkDown2 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md3 = st.number_input("MarkDown3 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md4 = st.number_input("MarkDown4 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md5 = st.number_input("MarkDown5 ($)", 0.0, value=0.0, step=100.0, format="%.0f")

    # ────────────────────────────────
    #   THIS IS REQUIRED
    # ────────────────────────────────
    submitted = st.form_submit_button("Predict Weekly Sales", type="primary", use_container_width=True)


# ────────────────────────────────────────────────
# Prediction
# ────────────────────────────────────────────────
if submitted:
    with st.spinner("Predicting..."):

        total_md = md1 + md2 + md3 + md4 + md5

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
            'Markdown_Group': '0',  # safe value – change to 'Low'/'Medium' if needed
        }

        X_input = pd.DataFrame([row])

        # Fix categorical dtype
        if 'Markdown_Group' in X_input.columns:
            X_input['Markdown_Group'] = X_input['Markdown_Group'].astype('category')

        # Add binned columns
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

        # Align columns
        try:
            expected = model.feature_names_in_
            X_input = X_input.reindex(columns=expected, fill_value=0)
            if 'Markdown_Group' in X_input.columns:
                X_input['Markdown_Group'] = X_input['Markdown_Group'].astype('category')
        except AttributeError:
            pass

        try:
            log_y = model.predict(X_input)[0]
            dollars = np.expm1(log_y)

            st.success("Prediction ready")
            st.metric("Predicted Weekly Sales", f"${dollars:,.0f}")
            st.metric("Log prediction", f"{log_y:.4f}")
            st.caption(f"Store {store} • Dept {dept} • {week_start:%Y-%m-%d}")

        except Exception as e:
            st.error(f"Prediction failed: {type(e).__name__} – {e}")
            st.info("Common fix: change 'Markdown_Group' to 'Low', 'Medium', 'High', '0', etc.")


with st.expander("Model info"):
    st.markdown("""
    - Model: XGBoost (Milestone 4)
    - Target: log₁ₚ(Weekly_Sales)
    - Approx performance: MAE $7k–$9k, R² 0.93–0.96
    """)
