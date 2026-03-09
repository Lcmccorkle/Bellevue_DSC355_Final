# =============================================================================
# PART 1: IMPORTS & SETUP
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# PART 2: DATA & MODEL LOADING (Cached)
# =============================================================================
@st.cache_resource
def load_model_and_reference_data():
    """
    Loads the trained XGBoost model and (if available) the reference dataset
    used for realistic dropdown options and default values.
    """
    # ── Load the model ───────────────────────────────────────────────────────
    model = None
    try:
        model = joblib.load('walmart_xgb_model.joblib')
    except FileNotFoundError:
        st.error("Model file 'walmart_xgb_model.joblib' not found.")
        st.error("→ Make sure the file is in the same folder as this script and correctly named.")
        st.stop()   # ← this stops execution → the rest of the function won't run

    # If we reach here → model was successfully loaded
    # ── Load reference data (try to combine the 4 part files) ────────────────
    df_ref = None
    part_files = [
        'engineered_walmart_data_Part1.csv',
        'engineered_walmart_data_Part2.csv',
        'engineered_walmart_data_Part3.csv',
        'engineered_walmart_data_Part4.csv'
    ]

    try:
        df_parts = [pd.read_csv(f, parse_dates=['Date']) for f in part_files]
        df_ref = pd.concat(df_parts, ignore_index=True)
        # Optional: show success message (can be removed in production)
        # st.success("Loaded all data parts successfully", icon="✅")
    except FileNotFoundError as e:
        st.warning(f"Could not load one or more data parts: {e}")
        st.info("Using fallback/default values for dropdowns and inputs.")
    except Exception as e:
        st.error(f"Unexpected error while loading data: {e}")

    return model, df_ref


# Load once and reuse
model, df_ref = load_model_and_reference_data()


# =============================================================================
# PART 3: PREPARE DROPDOWN CHOICES & DEFAULT VALUES
# =============================================================================
if df_ref is not None:
    stores = sorted(df_ref['Store'].unique().astype(int).tolist())
    depts  = sorted(df_ref['Dept'].unique().astype(int).tolist())
    types  = ['A', 'B', 'C']  # adjust if you have actual 'Type' column
else:
    # Fallback values when data is missing
    stores = list(range(1, 46))
    depts  = list(range(1, 100))
    types  = ['A', 'B', 'C']

# Reasonable default values
DEFAULTS = {
    'size': 140000,
    'temperature': 60.0,
    'fuel_price': 3.3,
    'unemployment': 7.0,
    'cpi': 170.0,
}


# =============================================================================
# PART 4: STREAMLIT USER INTERFACE & PREDICTION LOGIC
# =============================================================================
st.title("Walmart Weekly Sales Forecaster")
st.markdown("Predict weekly department-level sales using the XGBoost model from Milestone 4.")

st.info("Most important features (based on SHAP): Store, Dept, Size, IsHoliday, Temperature, Markdowns, Type, CPI, Unemployment")

# ── Input Form ───────────────────────────────────────────────────────────────
with st.form("sales_prediction_form"):

    # Store & Department
    st.subheader("Store & Department")
    col1, col2 = st.columns(2)
    with col1:
        store = st.selectbox("Store", options=stores, index=0)
    with col2:
        dept = st.selectbox("Department", options=depts, index=0)

    # Date & Holiday
    st.subheader("Date & Holiday")
    col3, col4 = st.columns(2)
    with col3:
        pred_date = st.date_input("Week start date", value=datetime(2012, 9, 1))
    with col4:
        is_holiday = st.checkbox("Holiday week?", value=False)

    # Store & Economic features
    st.subheader("Store & Economic Features")
    col5, col6, col7 = st.columns(3)
    with col5:
        size = st.number_input("Store Size (sq ft)", 30000, 220000, DEFAULTS['size'], step=5000)
        store_type = st.selectbox("Store Type", options=types, index=0)
    with col6:
        temperature = st.number_input("Temperature (°F)", -20.0, 110.0, DEFAULTS['temperature'], step=1.0)
        fuel_price  = st.number_input("Fuel Price ($/gal)", 1.5, 5.0, DEFAULTS['fuel_price'], step=0.1)
    with col7:
        unemployment = st.number_input("Unemployment Rate (%)", 3.0, 12.0, DEFAULTS['unemployment'], step=0.1)
        cpi = st.number_input("CPI", 120.0, 230.0, DEFAULTS['cpi'], step=0.1)

    # Promotions
    st.subheader("Promotions (MarkDowns)")
    md1 = st.number_input("MarkDown1 ($)", min_value=0.0, value=0.0, step=100.0, format="%.0f")
    md2 = st.number_input("MarkDown2 ($)", min_value=0.0, value=0.0, step=100.0, format="%.0f")
    md3 = st.number_input("MarkDown3 ($)", min_value=0.0, value=0.0, step=100.0, format="%.0f")
    md4 = st.number_input("MarkDown4 ($)", min_value=0.0, value=0.0, step=100.0, format="%.0f")
    md5 = st.number_input("MarkDown5 ($)", min_value=0.0, value=0.0, step=100.0, format="%.0f")

    submit = st.form_submit_button("Predict Weekly Sales")


# ── Prediction ───────────────────────────────────────────────────────────────
if submit:
    # Build input dictionary with features the model expects
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
        # Add more engineered features here if your model was trained with them
        # 'Total_MarkDown': md1 + md2 + md3 + md4 + md5,
        # 'Holiday_x_TotalMarkdown': is_holiday * (md1 + md2 + md3 + md4 + md5),
    }

    input_df = pd.DataFrame([input_dict])

    # Try to match the exact column order the model was trained on
    try:
        expected_cols = model.feature_names_in_
        input_df = input_df[expected_cols]
    except AttributeError:
        st.warning("Could not read model.feature_names_in_ — hoping column order matches.")

    # Make prediction
    log_pred = model.predict(input_df)[0]
    dollar_pred = np.expm1(log_pred)  # if you used np.log1p → use expm1
    # If you used np.log(Weekly_Sales) instead → change to np.exp(log_pred)

    # Show result
    st.success("Prediction ready!")
    st.metric("Predicted Weekly Sales", f"${dollar_pred:,.0f}")
    st.metric("(Log-scale internal value)", f"{log_pred:.4f}")

    st.caption(f"Store {store} • Dept {dept} • Week of {pred_date:%Y-%m-%d}")


# ── Model information ────────────────────────────────────────────────────────
with st.expander("Model Summary (Milestone 4)"):
    st.markdown("""
    - **Model**: XGBoost Regressor (tuned with RandomizedSearchCV + early stopping)
    - **Target**: log(Weekly_Sales + 1) → transformed back to dollars
    - **Approximate test performance**: MAE $7k–$9k, R² 0.93–0.96
    - **Limitations**: Trained on data up to ~Oct 2012
    """)

