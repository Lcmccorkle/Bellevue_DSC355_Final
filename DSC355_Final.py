# DSC355_Final.py
# Walmart Weekly Sales Forecaster

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ────────────────────────────────────────────────
#  Configuration & Constants
# ────────────────────────────────────────────────

MODEL_PATH = 'walmart_xgb_model.joblib'
PART_FILES = [
    'engineered_walmart_data_Part1.csv',
    'engineered_walmart_data_Part2.csv',
    'engineered_walmart_data_Part3.csv',
    'engineered_walmart_data_Part4.csv'
]

DEFAULTS = {
    'size': 140000,
    'temperature': 60.0,
    'fuel_price': 3.3,
    'unemployment': 7.0,
    'cpi': 170.0,
}

CSV_READ_KWARGS = {
    'parse_dates': ['Date'],
    'encoding': 'utf-8',
    'encoding_errors': 'replace',       # most forgiving option
    # 'encoding': 'cp1252',             # uncomment if utf-8 fails
    # 'encoding': 'latin1',
}


# ────────────────────────────────────────────────
#  Load model & reference data
# ────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # Load model
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}")
        st.error("Please place the trained model file in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {type(e).__name__}: {e}")
        st.stop()

    # Load reference data (multiple parts)
    df_ref = None
    try:
        parts = []
        for path in PART_FILES:
            df_part = pd.read_csv(path, **CSV_READ_KWARGS)
            parts.append(df_part)
        df_ref = pd.concat(parts, ignore_index=True)
        st.session_state.data_loaded = True
    except Exception as e:
        st.warning(f"Could not load all data parts: {e}")
        st.info("Using fallback values for dropdowns and ranges.")

    return model, df_ref


model, df_ref = load_resources()


# ────────────────────────────────────────────────
#  Prepare choices & defaults
# ────────────────────────────────────────────────
if df_ref is not None and 'Store' in df_ref.columns:
    stores = sorted(df_ref['Store'].unique().astype(int))
    depts  = sorted(df_ref['Dept'].unique().astype(int))
    types  = sorted(df_ref.get('Type', pd.Series(['A', 'B', 'C'])).unique())
else:
    stores = list(range(1, 46))
    depts  = list(range(1, 100))
    types  = ['A', 'B', 'C']


# ────────────────────────────────────────────────
#  UI
# ────────────────────────────────────────────────
st.title("Walmart Weekly Sales Forecaster")
st.caption("Predict department-level weekly sales (Milestone 4 XGBoost model)")

st.info("Most influential features (from SHAP): Store, Dept, Size, IsHoliday, Temperature, Total Markdown, Type, CPI, Unemployment")

with st.form("sales_form", clear_on_submit=False):

    # ── Store & Dept ───────────────────────────────
    st.subheader("Store & Department")
    c1, c2 = st.columns(2)
    with c1:
        store = st.selectbox("Store", stores, index=0)
    with c2:
        dept = st.selectbox("Department", depts, index=0)

    # ── Date & Holiday ─────────────────────────────
    st.subheader("Date & Holiday")
    c3, c4 = st.columns(2)
    with c3:
        week_start = st.date_input("Week start date", datetime(2012, 9, 1))
    with c4:
        is_holiday = st.checkbox("Holiday week", False)

    # ── Store & Economic ───────────────────────────
    st.subheader("Store & Economic Features")
    c5, c6, c7 = st.columns(3)
    with c5:
        size = st.number_input("Store Size (sq ft)", 30000, 220000, DEFAULTS['size'], step=5000)
        store_type = st.selectbox("Store Type", types, index=0)
    with c6:
        temperature = st.number_input("Temperature (°F)", -20.0, 110.0, DEFAULTS['temperature'], step=1.0)
        fuel_price = st.number_input("Fuel Price ($/gal)", 1.5, 5.0, DEFAULTS['fuel_price'], step=0.1)
    with c7:
        unemployment = st.number_input("Unemployment Rate (%)", 3.0, 12.0, DEFAULTS['unemployment'], step=0.1)
        cpi = st.number_input("CPI", 120.0, 230.0, DEFAULTS['cpi'], step=0.1)

    # ── Markdowns ──────────────────────────────────
    st.subheader("Promotions (MarkDowns)")
    md1 = st.number_input("MarkDown1 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md2 = st.number_input("MarkDown2 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md3 = st.number_input("MarkDown3 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md4 = st.number_input("MarkDown4 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md5 = st.number_input("MarkDown5 ($)", 0.0, value=0.0, step=100.0, format="%.0f")

    predict_btn = st.form_submit_button("Get Prediction", type="primary", use_container_width=True)


# ────────────────────────────────────────────────
#  Prediction
# ────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Preparing features and predicting..."):

        total_markdown = md1 + md2 + md3 + md4 + md5

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
            'Total_MarkDown': total_markdown,
            'Holiday_x_TotalMarkdown': int(is_holiday) * total_markdown,
            # Add more engineered features here when you know what is missing
        }

        X_input = pd.DataFrame([row])

        # Try to align columns
        try:
            expected = model.feature_names_in_
            available = [c for c in expected if c in X_input.columns]
            missing = [c for c in expected if c not in X_input.columns]

            if missing:
                st.warning(f"Missing {len(missing)} expected columns (using {len(available)}). "
                           f"Examples of missing: {missing[:4]}")

            if not available:
                st.error("No matching columns found. Cannot make prediction.")
                st.stop()

            X_input = X_input[available]

        except AttributeError:
            st.info("Model has no feature_names_in_ attribute – using current columns only.")

        # Predict
        try:
            log_y = model.predict(X_input)[0]
            y_dollars = np.expm1(log_y)   # assuming log1p was used

            st.success("Prediction ready")
            st.metric("Predicted Weekly Sales", f"${y_dollars:,.0f}")
            st.metric("Log prediction (internal)", f"{log_y:.4f}")
            st.caption(f"Store {store} • Dept {dept} • Week starting {week_start:%Y-%m-%d}")

        except Exception as e:
            st.error(f"Prediction failed: {type(e).__name__}: {e}")
            st.info("Most common causes: missing columns or incorrect data types.")


# ────────────────────────────────────────────────
#  Footer / Info
# ────────────────────────────────────────────────
with st.expander("Model information"):
    st.markdown("""
    - **Model**: XGBoost regressor (tuned with RandomizedSearchCV + early stopping)
    - **Target**: log₁ₚ(Weekly_Sales)
    - **Approximate performance** (test set): MAE $7,000–$9,000, R² 0.93–0.96
    - **Training period**: up to ~October 2012
    - **Limitations**: extreme holiday values and post-2012 changes not captured
    """)
