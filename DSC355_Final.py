# DSC355_Final.py
# Walmart Weekly Sales Forecaster – Clean & Robust Version

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

MODEL_FILE = 'walmart_xgb_model.joblib'

PART_FILES = [
    'engineered_walmart_data_Part1.csv',
    'engineered_walmart_data_Part2.csv',
    'engineered_walmart_data_Part3.csv',
    'engineered_walmart_data_Part4.csv'
]

# Try these encodings in order
ENCODING_ORDER = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1', 'utf-8-sig']

# Default values for missing features
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
    'Markdown_Group': '0',          # safe value - change to 'Low'/'Medium' if needed
    'Temp_Category_Hot': 0,
    'Temp_Category_Mild': 1,
    'Unemployment_Category_Low': 0,
    'Unemployment_Category_Medium': 1,
    'Size_Category_Medium': 1,
    'Size_Category_Small': 0,
    'FuelPrice_Category_Low': 0,
    'FuelPrice_Category_Medium': 1,
    'CPI_Category_Low': 0,
    'CPI_Category_Medium': 1,
}


# ────────────────────────────────────────────────
# Load model & reference data
# ────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Cannot load model: {type(e).__name__} – {e}")
        st.stop()

    df_ref = None
    parts = []

    for path in PART_FILES:
        loaded = False
        for enc in ENCODING_ORDER:
            try:
                df_part = pd.read_csv(
                    path,
                    parse_dates=['Date'],
                    encoding=enc,
                    encoding_errors='replace',
                    low_memory=False
                )
                parts.append(df_part)
                loaded = True
                break
            except:
                continue

        if not loaded:
            st.warning(f"Could not load {path}")

    if parts:
        df_ref = pd.concat(parts, ignore_index=True)

    return model, df_ref


model, df_ref = load_resources()


# ────────────────────────────────────────────────
# Choices for dropdowns
# ────────────────────────────────────────────────
if df_ref is not None and 'Store' in df_ref.columns:
    stores = sorted(df_ref['Store'].astype(int).unique())
    depts  = sorted(df_ref['Dept'].astype(int).unique())
    types  = ['A', 'B', 'C']
else:
    stores = list(range(1, 46))
    depts  = list(range(1, 100))
    types  = ['A', 'B', 'C']


# ────────────────────────────────────────────────
# MAIN UI
# ────────────────────────────────────────────────
st.title("Walmart Weekly Sales Forecaster")
st.caption("XGBoost model from Milestone 4")

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

    # ─────────────── REQUIRED ───────────────
    submitted = st.form_submit_button("Predict Weekly Sales", type="primary", use_container_width=True)


# ────────────────────────────────────────────────
# Prediction logic
# ────────────────────────────────────────────────
if submitted:
    with st.spinner("Running prediction..."):

        total_md = md1 + md2 + md3 + md4 + md5

        # Build input row
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
            'Markdown_Group': '0',  # <--- Safe value that exists in training
        }

        X_input = pd.DataFrame([row])

        # Force categorical dtype for known categoricals
        if 'Markdown_Group' in X_input.columns:
            X_input['Markdown_Group'] = X_input['Markdown_Group'].astype('category')

        # Add binned columns (simple rules)
        X_input['Temp_Category_Hot']     = 1 if temperature > 80 else 0
        X_input['Temp_Category_Mild']    = 1 if 50 <= temperature <= 80 else 0
        X_input['Unemployment_Category_Low']    = 1 if unemployment < 6 else 0
        X_input['Unemployment_Category_Medium'] = 1 if 6 <= unemployment <= 8 else 0
        X_input['Size_Category_Medium']  = 1 if 100000 <= size <= 180000 else 0
        X_input['Size_Category_Small']   = 1 if size < 100000 else 0
        X_input['FuelPrice_Category_Low']    = 1 if fuel_price < 3.0 else 0
        X_input['FuelPrice_Category_Medium'] = 1 if 3.0 <= fuel_price <= 4.0 else 0
        X_input['CPI_Category_Low']      = 1 if cpi < 150 else 0
        X_input['CPI_Category_Medium']   = 1 if 150 <= cpi <= 190 else 0

        # Align with model's expected columns
        try:
            expected_cols = model.feature_names_in_
            X_input = X_input.reindex(columns=expected_cols, fill_value=0)

            # Re-apply category dtype after fill
            if 'Markdown_Group' in X_input.columns:
                X_input['Markdown_Group'] = X_input['Markdown_Group'].astype('category')

        except AttributeError:
            st.info("Model does not expose feature names – using input as-is")

        # Make prediction
        try:
            log_pred = model.predict(X_input)[0]
            dollar_pred = np.expm1(log_pred)

            st.success("Prediction ready")
            st.metric("Predicted Weekly Sales", f"${dollar_pred:,.0f}")
            st.metric("Log-scale value (internal)", f"{log_pred:.4f}")
            st.caption(f"Store {store} • Dept {dept} • Week of {week_start:%Y-%m-%d}")

        except Exception as e:
            st.error(f"Prediction failed: {type(e).__name__} – {e}")
            if "category not in the training set" in str(e):
                st.info("Try changing 'Markdown_Group' to 'Low', 'Medium', 'High', '0', or another value from training data")


# ────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────
with st.expander("Model information"):
    st.markdown("""
    - Model: XGBoost Regressor (Milestone 4)
    - Target: log₁ₚ(Weekly_Sales)
    - Approx test performance: MAE $7k–$9k, R² 0.93–0.96
    - Some engineered features use defaults when not provided
    """)
