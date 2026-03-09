# =============================================================================
# Walmart Weekly Sales Forecaster – Streamlit App
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: LOAD MODEL & REFERENCE DATA
# =============================================================================
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load('walmart_xgb_model.joblib')
    except FileNotFoundError:
        st.error("Model file 'walmart_xgb_model.joblib' not found.")
        st.error("Make sure the file is in the same folder as this script.")
        st.stop()

    df_ref = None
    part_files = [
        'engineered_walmart_data_Part1.csv',
        'engineered_walmart_data_Part2.csv',
        'engineered_walmart_data_Part3.csv',
        'engineered_walmart_data_Part4.csv'
    ]

    try:
        parts = []
        for path in part_files:
            parts.append(pd.read_csv(path, parse_dates=['Date'], encoding='cp1252'))
        df_ref = pd.concat(parts, ignore_index=True)
    except Exception as e:
        st.warning(f"Could not load data parts: {e}")
        st.info("Using fallback defaults for dropdowns and ranges.")

    return model, df_ref


model, df_ref = load_model_and_data()


# =============================================================================
# PART 2: DROPDOWNS & DEFAULTS
# =============================================================================
if df_ref is not None:
    stores = sorted(df_ref['Store'].unique().astype(int))
    depts  = sorted(df_ref['Dept'].unique().astype(int))
    types  = ['A', 'B', 'C']
else:
    stores = list(range(1, 46))
    depts  = list(range(1, 100))
    types  = ['A', 'B', 'C']

DEFAULTS = {
    'size': 140000,
    'temp': 60.0,
    'fuel': 3.3,
    'unemp': 7.0,
    'cpi': 170.0
}


# =============================================================================
# PART 3: USER INTERFACE
# =============================================================================
st.title("Walmart Weekly Sales Forecaster")
st.markdown("Predict department-level weekly sales using the XGBoost model from Milestone 4.")

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
        pred_date = st.date_input("Week start", datetime(2012, 9, 1))
    with col4:
        is_holiday = st.checkbox("Holiday week?", False)

    st.subheader("Store & Economic Features")
    col5, col6, col7 = st.columns(3)
    with col5:
        size = st.number_input("Store Size (sq ft)", 30000, 220000, DEFAULTS['size'], step=5000)
        store_type = st.selectbox("Store Type", types, index=0)
    with col6:
        temperature = st.number_input("Temperature (°F)", -20.0, 110.0, DEFAULTS['temp'], step=1.0)
        fuel_price  = st.number_input("Fuel Price ($/gal)", 1.5, 5.0, DEFAULTS['fuel'], step=0.1)
    with col7:
        unemployment = st.number_input("Unemployment (%)", 3.0, 12.0, DEFAULTS['unemp'], step=0.1)
        cpi = st.number_input("CPI", 120.0, 230.0, DEFAULTS['cpi'], step=0.1)

    st.subheader("MarkDowns")
    md1 = st.number_input("MarkDown1 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md2 = st.number_input("MarkDown2 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md3 = st.number_input("MarkDown3 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md4 = st.number_input("MarkDown4 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md5 = st.number_input("MarkDown5 ($)", 0.0, value=0.0, step=100.0, format="%.0f")

    submitted = st.form_submit_button("Predict Weekly Sales")


# =============================================================================
# PART 4: PREDICTION LOGIC
# =============================================================================
if submitted:
    with st.spinner("Preparing input features..."):
        total_md = md1 + md2 + md3 + md4 + md5

        input_dict = {
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
            'Year': pred_date.year,
            'Month': pred_date.month,
            'Week': pred_date.isocalendar()[1],
            'Quarter': (pred_date.month - 1) // 3 + 1,
            'DayOfWeek': pred_date.weekday(),
            'Total_MarkDown': total_md,
            'Holiday_x_TotalMarkdown': int(is_holiday) * total_md,
            # Add more derived features / bins if needed
        }

        input_df = pd.DataFrame([input_dict])

    # Try to align columns with model expectations
    try:
        expected = model.feature_names_in_

        # Only use columns the model knows
        available = [c for c in expected if c in input_df.columns]
        missing = [c for c in expected if c not in input_df.columns]

        if missing:
            st.warning(f"Missing {len(missing)} expected columns. "
                       f"Using {len(available)} available columns. "
                       f"Missing example: {missing[:3]}")

        input_df = input_df[available]

    except AttributeError:
        st.info("Model has no feature_names_in_ attribute – using current columns only.")

    # Make prediction
    try:
        log_pred = model.predict(input_df)[0]
        dollar_pred = np.expm1(log_pred)   # if target was log1p

        st.success("Prediction successful!")
        st.metric("Predicted Weekly Sales", f"${dollar_pred:,.0f}")
        st.metric("Log prediction (internal)", f"{log_pred:.4f}")
        st.caption(f"Store {store} • Dept {dept} • Week of {pred_date:%Y-%m-%d}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Check that all required features are provided and have correct types.")


# =============================================================================
# MODEL INFO
# =============================================================================
with st.expander("About this model"):
    st.markdown("""
    - **Model**: XGBoost Regressor (Milestone 4)
    - **Target**: log(Weekly_Sales + 1)
    - **Approx performance**: MAE $7k–$9k, R² ~0.93–0.96
    - **Data**: Up to ~Oct 2012
    """)
