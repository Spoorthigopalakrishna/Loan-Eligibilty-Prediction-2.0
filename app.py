"""
app.py — Loan Eligibility Predictor (Phase 5: Professional Dashboard)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Page Config
st.set_page_config(
    page_title="Loan Intelligence Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS for a clean, wide dashboard
st.markdown("""
    <style>
    /* Hide Sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    .main {
        background-color: #f8fafc;
        color: #0f172a;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] > div {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] > div {
        color: #64748b !important;
    }
    .badge {
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 9999px;
        display: inline-block;
        font-size: 0.875rem;
    }
    .badge-approved {
        color: #065f46;
        background-color: #dcfce7;
    }
    .badge-rejected {
        color: #991b1b;
        background-color: #fee2e2;
    }
    .header-text {
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    try:
        model         = joblib.load(os.path.join(models_dir, 'loan_model.pkl'))
        explainer     = joblib.load(os.path.join(models_dir, 'explainer.pkl'))
        feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
        encoders      = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
        metrics       = joblib.load(os.path.join(models_dir, 'metrics.pkl'))
        return model, explainer, feature_names, encoders, metrics, None
    except Exception as e:
        files = ['loan_model.pkl', 'explainer.pkl', 'feature_names.pkl', 'label_encoders.pkl', 'metrics.pkl']
        missing = [f for f in files if not os.path.exists(os.path.join(models_dir, f))]
        if missing:
            return None, None, None, None, None, f"Missing files in {models_dir}: {', '.join(missing)}"
        return None, None, None, None, None, f"Error loading models: {str(e)}"

model, explainer, feature_names, encoders, metrics, error_msg = load_artifacts()

if model is None:
    st.error(f"Model initialization failed: {error_msg}")
    st.stop()

# ─────────────────────────────────────────────
# Header Section
# ─────────────────────────────────────────────
st.title("🏦 Loan Intelligence Dashboard")
st.markdown("<div style='text-align: center; margin-bottom: 2rem;'>Professional-grade credit risk assessment powered by XGBoost & SHAP.</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────
with st.container(border=True):
    st.subheader("👤 Applicant Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ("Male", "Female"))
        married = st.selectbox("Married", ("Yes", "No"))
        dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))
        education = st.selectbox("Education", ("Graduate", "Not Graduate"))

    with col2:
        self_employed = st.selectbox("Self Employed", ("Yes", "No"))
        applicant_income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
        coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0)
        property_area = st.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))

    with col3:
        loan_amount = st.number_input("Loan Amount ($k)", min_value=0, value=150)
        loan_term = st.selectbox("Term (months)", (360, 120, 180, 240, 300, 480))
        credit_history = st.selectbox("Credit History", (1.0, 0.0), help="1.0: Good, 0.0: Delinquent")
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔍 Assess Eligibility", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# Logic & Inference
# ─────────────────────────────────────────────
def get_input_df():
    data = {
        'Gender': gender, 'Married': married, 'Dependents': dependents,
        'Education': education, 'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income, 'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount, 'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history, 'Property_Area': property_area,
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
    return df[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0)

input_df = get_input_df()
processed_input = preprocess_input(input_df)

def get_shap_values(explainer, processed_input):
    shap_values = explainer.shap_values(processed_input)
    if isinstance(shap_values, list):
        vals = np.array(shap_values[1])
        base_val = float(np.array(explainer.expected_value).flat[1])
    else:
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            vals = shap_values[:, :, 1]
            base_val = float(np.array(explainer.expected_value).flat[1])
        else:
            vals = shap_values
            base_val = float(explainer.expected_value)
    return vals.flatten()[:len(feature_names)], base_val

# ─────────────────────────────────────────────
# Results Section
# ─────────────────────────────────────────────
if predict_button:
    prediction = model.predict(processed_input)
    probability = model.predict_proba(processed_input)
    approved = prediction[0] == 1
    confidence = max(probability[0]) * 100

    with st.container(border=True):
        st.subheader("📊 Decision Assessment")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if approved:
                st.markdown("<div class='badge badge-approved'>✅ ELIGIBLE</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='badge badge-rejected'>❌ INELIGIBLE</div>", unsafe_allow_html=True)
            st.metric("Model Confidence", f"{confidence:.1f}%")
        
        with res_col2:
            st.progress(float(confidence / 100))
            st.write(f"Eligibility probability: **{probability[0][1]*100:.1f}%**")
            st.caption("A higher percentage indicates a stronger financial profile for this specific loan.")

    # Explainability Section
    with st.container(border=True):
        st.subheader(f"🔍 Factor Analysis")
        st.write(f"The following factors had the most significant impact on the decision:")
        
        try:
            vals_1d, base_val = get_shap_values(explainer, processed_input)
            explanation = shap.Explanation(
                values=vals_1d, base_values=base_val,
                data=processed_input.iloc[0].values,
                feature_names=list(feature_names)
            )
            
            tab1, tab2 = st.tabs(["Decision Flow (Waterfall)", "Impact Table"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(explanation, show=False)
                st.pyplot(plt.gcf(), use_container_width=True)
                plt.clf()
                
            with tab2:
                shap_df = pd.DataFrame({
                    'Feature': list(feature_names),
                    'Impact Score': vals_1d,
                }).sort_values('Impact Score', ascending=False)
                shap_df['Contribution'] = shap_df['Impact Score'].apply(lambda x: '🟢 Positive' if x > 0 else '🔴 Negative')
                st.dataframe(shap_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Analysis failed: {e}")

# ─────────────────────────────────────────────
# End of Application
# ─────────────────────────────────────────────