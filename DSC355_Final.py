# DSC355_Final.py
# Walmart Weekly Sales Forecaster – with quick encoding fix

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

# Encoding that usually fixes 0x9d / invalid continuation byte errors
CSV_READ_KWARGS = {
    'parse_dates': ['Date'],
    'encoding': 'cp1252',              # ← Quick resolution: most common Windows encoding
    'encoding_errors': 'replace',      # replace invalid chars with �
    'low_memory': False
}


# ────────────────────────────────────────────────
#  Load model & reference data
# ────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        st.caption("Model loaded successfully")
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

    # Load data parts with fixed encoding
    df_ref = None
    loaded_parts = []

    for path in PART_FILES:
        try:
            df_part = pd.read_csv(path, **CSV_READ_KWARGS)
            loaded_parts.append(df_part)
            st.caption(f"Loaded: {path}")
        except UnicodeDecodeError:
            # Fallback for stubborn files
            st.warning(f"cp1252 failed on {path} → trying latin1")
            try:
                df_part = pd.read_csv(
                    path,
                    parse_dates=['Date'],
                    encoding='latin1',
                    encoding_errors='replace'
                )
                loaded_parts.append(df_part)
                st.caption(f"Loaded {path} using latin1 fallback")
            except Exception as e2:
                st.error(f"Could not load {path} even with fallback: {e2}")
        except Exception as e:
            st.error(f"Failed to read {path}: {e}")

    if loaded_parts:
        try:
            df_ref = pd.concat(loaded_parts, ignore_index=True)
            st.success(f"Loaded {len(loaded_parts)} / {len(PART_FILES)} data parts")
        except Exception as e:
            st.error(f"Concatenation failed: {e}")
    else:
        st.warning("No data parts loaded → dropdowns will use fallback ranges")

    return model, df_ref


model, df_ref = load_resources()


# ────────────────────────────────────────────────
#  Choices & defaults
# ────────────────────────────────────────────────
if df_ref is not None and 'Store' in df_ref.columns:
    stores = sorted(df_ref['Store'].unique().astype(int))
    depts  = sorted(df_ref['Dept'].unique().astype(int))
    types  = ['A', 'B', 'C']  # fallback if Type missing
else:
    stores = list(range(1, 46))
    depts  = list(range(1, 100))
    types  = ['A', 'B', 'C']

DEFAULTS = {
    'size': 140000,
    'temperature': 60.0,
    'fuel_price': 3.3,
    'unemployment': 7.0,
    'cpi': 170.0
}


# ────────────────────────────────────────────────
#  Main UI
# ────────────────────────────────────────────────
st.title("Walmart Weekly Sales Forecaster")
st.markdown("Predict weekly department sales using the Milestone 4 XGBoost model.")

with st.form("prediction_form"):

    st.subheader("Store & Department")
    c1, c2 = st.columns(2)
    with c1:
        store = st.selectbox("Store", stores, index=0)
    with c2:
        dept = st.selectbox("Department", depts, index=0)

    st.subheader("Date & Holiday")
    c3, c4 = st.columns(2)
    with c3:
        week_start = st.date_input("Week start date", datetime(2012, 9, 1))
    with c4:
        is_holiday = st.checkbox("Holiday week?", False)

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

    st.subheader("MarkDowns")
    md1 = st.number_input("MarkDown1 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md2 = st.number_input("MarkDown2 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md3 = st.number_input("MarkDown3 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md4 = st.number_input("MarkDown4 ($)", 0.0, value=0.0, step=100.0, format="%.0f")
    md5 = st.number_input("MarkDown5 ($)", 0.0, value=0.0, step=100.0, format="%.0f")

    submitted = st.form_submit_button("Predict Weekly Sales", type="primary")


# ────────────────────────────────────────────────
#  Prediction
# ────────────────────────────────────────────────
if submitted:
    with st.spinner("Preparing input and predicting..."):

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
            # Add more derived / binned features if needed
        }

        X_input = pd.DataFrame([row])

        # Safe column alignment
        try:
            expected = model.feature_names_in_
            available = [c for c in expected if c in X_input.columns]
            missing = [c for c in expected if c not in X_input.columns]

            if missing:
                st.warning(f"Using {len(available)}/{len(expected)} columns. "
                           f"Missing: {missing[:5]}{'...' if len(missing)>5 else ''}")

            if not available:
                st.error("No matching columns found → cannot predict.")
                st.stop()

            X_input = X_input[available]

        except AttributeError:
            st.info("Model has no feature_names_in_ → using current columns as-is.")

        # Predict
        try:
            log_y = model.predict(X_input)[0]
            y_dollars = np.expm1(log_y)

            st.success("Prediction ready!")
            st.metric("Predicted Weekly Sales", f"${y_dollars:,.0f}")
            st.metric("Log prediction (internal)", f"{log_y:.4f}")
            st.caption(f"Store {store} • Dept {dept} • Week of {week_start:%Y-%m-%d}")

        except Exception as e:
            st.error(f"Prediction failed: {type(e).__name__} – {e}")


# ────────────────────────────────────────────────
#  Footer
# ────────────────────────────────────────────────
with st.expander("Model details"):
    st.markdown("""
    - **Model**: XGBoost (Milestone 4)
    - **Target**: log₁ₚ(Weekly_Sales)
    - **Approx test performance**: MAE $7k–$9k, R² 0.93–0.96
    - **Data cutoff**: ~Oct 2012
    """)
