import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Function to clean currency
def clean_currency(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace('$', '').replace('(', '-').replace(')', '').replace(',', '')
    try:
        return float(val)
    except:
        return np.nan

# Function to clean percent
def clean_percent(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace('%', '')
    try:
        return float(val) / 100
    except:
        return np.nan

# Load and prepare data
@st.cache_data
def load_and_train_model():
    df = pd.read_csv("eda_classification.csv")

    # Standardize month names
    month_map = {
        'sept.': 'Sep', 'sept': 'Sep', 'Dev': 'Dec', 'thurday': 'Thursday',
        'wed': 'Wednesday', 'thur': 'Thursday', 'tuesday': 'Tuesday',
        'wednesday': 'Wednesday', 'friday': 'Friday'
    }
    df['x1'] = df['x1'].replace(month_map)
    df['x14'] = df['x14'].replace(month_map)

    # Fix brand names
    brand_map = {'volkswagon': 'volkswagen', 'chrystler': 'chrysler'}
    df['x13'] = df['x13'].replace(brand_map)

    # Fix size
    df['x17'] = df['x17'].str.lower()

    # Parse x7 and x11
    df['x7_clean'] = df['x7'].apply(clean_currency)
    df['x11_clean'] = df['x11'].apply(clean_percent)

    # Drop rows with missing values
    df = df.dropna()

    # Features (using cleaned versions)
    numeric_features = ['x0', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7_clean', 'x8', 'x9', 'x10', 'x11_clean', 'x12', 'x15', 'x16']
    categorical_features = ['x1', 'x13', 'x14', 'x17']
    features = numeric_features + categorical_features

    X = df[features]
    y = df['y']

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Pipeline with tuned Logistic Regression (best from notebook)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(C=100, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42))])

    # Train on full data for deployment
    pipeline.fit(X, y)

    return pipeline, df

# Load model and data
pipeline, df = load_and_train_model()

# Streamlit App
st.title("Predictive Model Dashboard")

st.markdown("""
This app recreates a predictive model dashboard similar to the one described in *Hands-On Predictive Analytics with Python* Chapter 9, 
but using Streamlit instead of Dash, and deploying the classification model from the provided Week7 model. 
The model predicts the binary target 'y' based on input features. Note: Model performance is near-random.
""")

# Input form
with st.form("prediction_form"):
    st.header("Enter Feature Values")

    # Numerical inputs (use data min/max/mean for defaults)
    x0 = st.number_input("x0", min_value=float(df['x0'].min()), max_value=float(df['x0'].max()), value=float(df['x0'].mean()))
    x2 = st.number_input("x2", min_value=float(df['x2'].min()), max_value=float(df['x2'].max()), value=float(df['x2'].mean()))
    x3 = st.number_input("x3", min_value=float(df['x3'].min()), max_value=float(df['x3'].max()), value=float(df['x3'].mean()))
    x4 = st.number_input("x4", min_value=float(df['x4'].min()), max_value=float(df['x4'].max()), value=float(df['x4'].mean()))
    x5 = st.number_input("x5", min_value=float(df['x5'].min()), max_value=float(df['x5'].max()), value=float(df['x5'].mean()))
    x6 = st.number_input("x6", min_value=float(df['x6'].min()), max_value=float(df['x6'].max()), value=float(df['x6'].mean()))
    x7 = st.text_input("x7 (currency, e.g., '$1,234.56' or '($1,234.56)')", value="$0.00")
    x8 = st.number_input("x8", min_value=float(df['x8'].min()), max_value=float(df['x8'].max()), value=float(df['x8'].mean()))
    x9 = st.number_input("x9", min_value=float(df['x9'].min()), max_value=float(df['x9'].max()), value=float(df['x9'].mean()))
    x10 = st.number_input("x10", min_value=float(df['x10'].min()), max_value=float(df['x10'].max()), value=float(df['x10'].mean()))
    x11 = st.text_input("x11 (percent, e.g., '0.01%' or '-0.01%')", value="0.00%")
    x12 = st.number_input("x12", min_value=float(df['x12'].min()), max_value=float(df['x12'].max()), value=float(df['x12'].mean()))
    x15 = st.number_input("x15", min_value=float(df['x15'].min()), max_value=float(df['x15'].max()), value=float(df['x15'].mean()))
    x16 = st.number_input("x16", min_value=float(df['x16'].min()), max_value=float(df['x16'].max()), value=float(df['x16'].mean()))

    # Categorical inputs
    x1 = st.selectbox("x1 (month)", options=sorted(df['x1'].unique()))
    x13 = st.selectbox("x13 (brand)", options=sorted(df['x13'].unique()))
    x14 = st.selectbox("x14 (day)", options=sorted(df['x14'].unique()))
    x17 = st.selectbox("x17 (size)", options=sorted(df['x17'].unique()))

    submit = st.form_submit_button("Predict")

if submit:
    # Create input DataFrame
    input_data = {
        'x0': x0, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6,
        'x7': x7, 'x8': x8, 'x9': x9, 'x10': x10, 'x11': x11, 'x12': x12,
        'x15': x15, 'x16': x16, 'x1': x1, 'x13': x13, 'x14': x14, 'x17': x17
    }
    input_df = pd.DataFrame([input_data])

    # Apply cleaning
    input_df['x7_clean'] = input_df['x7'].apply(clean_currency)
    input_df['x11_clean'] = input_df['x11'].apply(clean_percent)

    # Select features for prediction (drop raw x7, x11)
    input_features = input_df[['x0', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7_clean', 'x8', 'x9', 'x10',
                               'x11_clean', 'x12', 'x15', 'x16', 'x1', 'x13', 'x14', 'x17']]

    # Predict
    prediction = pipeline.predict(input_features)[0]
    proba = pipeline.predict_proba(input_features)[0][1]  # Probability of class 1

    st.header("Prediction Result")
    st.write(f"Predicted y: **{prediction}**")
    st.write(f"Probability of y=1: **{proba:.2%}**")

# Add section for model info
with st.expander("Model Details"):
    st.markdown("""
    - **Model**: Tuned Logistic Regression (C=100, l2 penalty, lbfgs solver)
    - **Training**: Fit on full cleaned dataset (~9968 samples)
    - **Expected Accuracy**: ~50.7% (near random, as per notebook analysis)
    - For production, consider pickling the model instead of retraining on load.
    """)

