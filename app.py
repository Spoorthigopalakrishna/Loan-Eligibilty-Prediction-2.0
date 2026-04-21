"""
app.py — Loan Eligibility Predictor (Phase 4: Polished UI)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")

# Custom CSS for polished UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Ensure metric text is dark regardless of theme */
    div[data-testid="stMetricValue"] > div {
        color: #1f2937 !important;
    }
    div[data-testid="stMetricLabel"] > div {
        color: #4b5563 !important;
    }
    .badge {
        font-weight: bold;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
    }
    .badge-approved {
        color: #2ecc71;
        background-color: #e8f8f5;
    }
    .badge-rejected {
        color: #e74c3c;
        background-color: #fdedec;
    }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model         = joblib.load('models/loan_model.pkl')
        explainer     = joblib.load('models/explainer.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        encoders      = joblib.load('models/label_encoders.pkl')
        metrics       = joblib.load('models/metrics.pkl')
        return model, explainer, feature_names, encoders, metrics
    except FileNotFoundError:
        return None, None, None, None, None

model, explainer, feature_names, encoders, metrics = load_artifacts()

if model is None:
    st.error("Model files not found. Please run `python src/train_model.py` first.")
    st.stop()

# ─────────────────────────────────────────────
# Sidebar — Model Performance
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Dashboard Settings")
with st.sidebar.expander("📊 View Model Performance", expanded=False):
    if metrics:
        st.metric("Test Accuracy", f"{metrics['accuracy']*100:.1f}%")
        st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
        
        try:
            st.image('models/confusion_matrix.png', use_container_width=True)
        except:
            st.write(metrics['confusion_matrix'])
    else:
        st.warning("Metrics not found. Run training script.")

st.sidebar.markdown("---")
st.sidebar.info("Model: XGBoost with Class-Imbalance Handling")

# ─────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────
st.title("🏦 Loan Eligibility Predictor")
st.write("Determine customer eligibility for a loan and explain the model's decision.")

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
with st.container():
    st.subheader("📝 Customer Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ("Male", "Female"))
        married = st.selectbox("Married", ("Yes", "No"))
        dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))
        education = st.selectbox("Education", ("Graduate", "Not Graduate"))

    with col2:
        self_employed = st.selectbox("Self Employed", ("Yes", "No"))
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
        property_area = st.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))

    with col3:
        loan_amount = st.number_input("Loan Amount ($k)", min_value=0, value=150)
        loan_term = st.selectbox("Loan Term (months)", (360, 120, 180, 240, 300, 480))
        credit_history = st.selectbox("Credit History", (1.0, 0.0), help="1.0: Good, 0.0: Bad")
    
    predict_button = st.button("🔍 Run Prediction", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
def get_input_df():
    data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area,
    }
    df = pd.DataFrame(data, index=[0])
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Log_Loan_Amount'] = np.log1p(df['LoanAmount'])
    df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1e-4)
    df['EMI_to_Income_Ratio'] = df['EMI'] / ((df['Total_Income'] / 12) + 1e-4)
    return df

def preprocess_input(df):
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            df[col] = le.transform([val])[0] if val in le.classes_ else -1
    df = df[feature_names]
    return df.apply(pd.to_numeric, errors='coerce').fillna(0)

input_df = get_input_df()
processed_input = preprocess_input(input_df)

# ─────────────────────────────────────────────
# SHAP Helpers
# ─────────────────────────────────────────────
def get_shap_values(explainer, processed_input):
    shap_values = explainer.shap_values(processed_input)
    if isinstance(shap_values, list): # Multi-output
        vals = np.array(shap_values[1]) # Class 1 (Approved)
        base_val = float(np.array(explainer.expected_value).flat[1])
    else: # Binary output for TreeExplainer sometimes dim=2
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            vals = shap_values[:, :, 1]
            base_val = float(np.array(explainer.expected_value).flat[1])
        else:
            vals = shap_values
            base_val = float(explainer.expected_value)
    return vals.flatten()[:len(feature_names)], base_val

# ─────────────────────────────────────────────
# Output Panel
# ─────────────────────────────────────────────
if predict_button:
    prediction = model.predict(processed_input)
    probability = model.predict_proba(processed_input)
    approved = prediction[0] == 1
    confidence = max(probability[0]) * 100

    st.markdown("---")
    
    # 1. Results Badge & Confidence
    st.subheader("💡 Prediction Result")
    res_col1, res_col2 = st.columns([1, 3])
    
    with res_col1:
        if approved:
            st.markdown("<div class='badge badge-approved'>✅ APPROVED</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='badge badge-rejected'>❌ REJECTED</div>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence:.1f}%")
    
    with res_col2:
        # Cast to float to avoid float32 errors and clamp between 0.0 and 1.0
        prog_val = float(confidence / 100)
        st.progress(min(max(prog_val, 0.0), 1.0))
        st.write(f"Approval Prob: **{probability[0][1]*100:.1f}%** | Rejection Prob: **{probability[0][0]*100:.1f}%**")

    # 2. Local Explanation (Why?)
    st.markdown("---")
    st.subheader(f"🔍 Why was this loan {'Approved' if approved else 'Rejected'}?")
    
    try:
        vals_1d, base_val = get_shap_values(explainer, processed_input)
        explanation = shap.Explanation(
            values=vals_1d,
            base_values=base_val,
            data=processed_input.iloc[0].values,
            feature_names=list(feature_names)
        )
        
        tab1, tab2 = st.tabs(["Waterfall Chart", "Detailed Table"])
        
        with tab1:
            st.markdown(f"**Top reasons for this decision:**")
            fig, ax = plt.subplots()
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.clf()
            
        with tab2:
            # Create a clean impact table
            shap_df = pd.DataFrame({
                'Feature': list(feature_names),
                'Impact (SHAP)': vals_1d,
            }).sort_values('Impact (SHAP)', ascending=False)
            
            # Add dynamic coloring for the impact
            shap_df['Direction'] = shap_df['Impact (SHAP)'].apply(lambda x: '🟢 Positive' if x > 0 else '🔴 Negative')
            
            st.dataframe(shap_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Explanation failed: {e}")

    # 3. Global Context Section
    with st.expander("🌍 How the model works globally"):
        try:
            st.image('models/shap_summary_plot.png', caption='Global SHAP Summary')
        except:
            st.info("Global summary plot not found.")
else:
    st.info("Fill out the details and click 'Run Prediction' to see results.")