import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────
#  Cache model & reference data
# ───────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load('walmart_xgb_model.joblib')
    except FileNotFoundError:
        st.error("Model file 'walmart_xgb_model.joblib' not found. Please save it from your notebook.")
        st.stop()

    # Load original data just to get realistic choices / defaults
    try:
        df = pd.read_csv('engineered_walmart_data.csv', parse_dates=['Date'])
    except FileNotFoundError:
        st.warning("Could not load engineered_walmart_data.csv — using fallback defaults.")
        df = None

    return model, df


model, df_ref = load_model_and_data()

# ───────────────────────────────────────────────
#  Prepare choices for dropdowns
# ───────────────────────────────────────────────
if df_ref is not None:
    stores = sorted(df_ref['Store'].unique().astype(int).tolist())
    depts  = sorted(df_ref['Dept'].unique().astype(int).tolist())
    types  = sorted(df_ref['Type'].dropna().unique().tolist()) if 'Type' in df_ref else ['A', 'B', 'C']
else:
    stores = list(range(1, 46))
    depts  = list(range(1, 100))
    types  = ['A', 'B', 'C']

# Reasonable defaults / medians from your data description
default_size         = 140000       # approx median
default_temp         = 60.0
default_fuel         = 3.3
default_unemp        = 7.0
default_cpi          = 170.0
default_markdown_tot = 0.0

# ───────────────────────────────────────────────
#  Streamlit App
# ───────────────────────────────────────────────
st.title("Walmart Weekly Sales Forecaster")
st.markdown("""
This app uses the **XGBoost model** trained in Milestone 4 to predict **log(Weekly Sales)**  
(and converts it back to dollars) for a given store, department, and week conditions.
""")

st.info("Best performing features according to SHAP: Store, Dept, Size, IsHoliday, Temperature, Total_MarkDown, Type, Unemployment, CPI, …")

# ── Input Form ───────────────────────────────────
with st.form("sales_prediction_form"):
    st.subheader("Store & Department")

    col1, col2 = st.columns(2)
    with col1:
        store = st.selectbox("Store", options=stores, index=0)
    with col2:
        dept = st.selectbox("Department", options=depts, index=0)

    st.subheader("Date & Holiday")

    col3, col4 = st.columns(2)
    with col3:
        pred_date = st.date_input("Prediction week start date", value=datetime(2012, 9, 1))
    with col4:
        is_holiday = st.checkbox("Is Holiday week?", value=False)

    st.subheader("Store & Economic Features")

    col5, col6, col7 = st.columns(3)
    with col5:
        size = st.number_input("Store Size (sq ft)", min_value=30000, max_value=220000, value=default_size, step=5000)
        store_type = st.selectbox("Store Type", options=types, index=0)
    with col6:
        temperature = st.number_input("Temperature (°F)", min_value=-20.0, max_value=110.0, value=default_temp, step=1.0)
        fuel_price = st.number_input("Fuel Price ($/gal)", min_value=1.5, max_value=5.0, value=default_fuel, step=0.1)
    with col7:
        unemployment = st.number_input("Unemployment Rate (%)", min_value=3.0, max_value=12.0, value=default_unemp, step=0.1)
        cpi = st.number_input("CPI", min_value=120.0, max_value=230.0, value=default_cpi, step=0.1)

    st.subheader("Promotions (MarkDowns)")

    md1 = st.number_input("MarkDown1 ($)", min_value=0.0, value=0.0, step=100.0, format="%.0f")
    md2 = st.number_input("MarkDown2 ($)", min_value=0.0, value=0.0, step=100.0, format="%.0f")
    md3 = st.number_input("MarkDown3 ($)", min_value=0.0, value=0.0, step=100.0, format="%.0f")
    md4 = st.number_input("MarkDown4 ($)", min_value=0.0, value=0.0, step=100.0, format="%.0f")
    md5 = st.number_input("MarkDown5 ($)", min_value=0.0, value=0.0, step=100.0, format="%.0f")

    submit = st.form_submit_button("Predict Weekly Sales")

# ── Prediction ───────────────────────────────────
if submit:
    # Create input row matching training feature names
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
        # Derived / categorical (you may need to match exactly what you used)
        'Type_B': 1 if store_type == 'B' else 0,
        'Type_C': 1 if store_type == 'C' else 0,
        # Add more dummies / derived features if they were in your final X
        # e.g. 'Total_MarkDown': md1+md2+md3+md4+md5,
        # 'Holiday_x_TotalMarkdown': is_holiday * (md1+md2+md3+md4+md5),
        # Year / Month / Week / DayOfWeek can be derived from pred_date
    }

    # Quick derivation of time features (adjust column names to match your training set)
    input_dict['Year']      = pred_date.year
    input_dict['Month']     = pred_date.month
    input_dict['Week']      = pred_date.isocalendar()[1]
    input_dict['DayOfWeek'] = pred_date.weekday()
    input_dict['Quarter']   = (pred_date.month - 1) // 3 + 1

    # Create DataFrame (one row)
    input_df = pd.DataFrame([input_dict])

    # Re-order columns to match exactly what the model was trained on
    # (critical!)
    try:
        model_features = model.feature_names_in_   # XGBoost ≥ 1.4
        input_df = input_df[model_features]
    except AttributeError:
        # fallback — you can hard-code the order if needed
        st.warning("Could not read model.feature_names_in_ — assuming column order matches.")

    # Predict (log scale)
    log_pred = model.predict(input_df)[0]

    # Back to dollars
    dollar_pred = np.expm1(log_pred)          # if you used log1p
    # or: dollar_pred = np.exp(log_pred)      # if you used plain np.log

    # Show result
    st.success("Prediction ready!")
    st.metric("Predicted Weekly Sales", f"${dollar_pred:,.0f}")
    st.metric("Log-scale prediction (internal)", f"{log_pred:.4f}")

    st.caption(f"For Store {store} • Dept {dept} • Week of {pred_date:%Y-%m-%d}")

# ── Model Information ────────────────────────────
with st.expander("Model & Performance Summary (Milestone 4)"):
    st.markdown("""
    - **Model**: XGBoost Regressor (tuned with RandomizedSearchCV + early stopping)
    - **Target**: log(Weekly_Sales + 1) → back-transformed to dollars
    - **Test MAE** (approx): $7,000 – $9,000
    - **R²** (approx): 0.93 – 0.96
    - **Most important features** (from SHAP): Store, Dept, Size, IsHoliday, Temperature, Total Markdown, Type, Unemployment, CPI
    - **Limitations**: No future data after Oct 2012; holiday extremes may still be under/over-predicted
    """)

