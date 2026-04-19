import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")

st.title("🏦 Loan Eligibility Predictor")
st.markdown("Enter customer details to predict loan eligibility and understand the reasoning behind the decision.")

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('models/loan_model.pkl')
        explainer = joblib.load('models/explainer.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, explainer, feature_names
    except FileNotFoundError:
        return None, None, None

model, explainer, feature_names = load_artifacts()

if model is None:
    st.error("Model files not found. Please run `src/train_model.py` first.")
    st.stop()

# Sidebar for inputs
st.sidebar.header("Customer Information")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    married = st.sidebar.selectbox("Married", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Dependents", ("0", "1", "2", "3+"))
    education = st.sidebar.selectbox("Education", ("Graduate", "Not Graduate"))
    self_employed = st.sidebar.selectbox("Self Employed", ("Yes", "No"))
    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=150)
    loan_term = st.sidebar.selectbox("Loan Amount Term (months)", (360, 120, 180, 240, 300, 480))
    credit_history = st.sidebar.selectbox("Credit History", (1.0, 0.0))
    property_area = st.sidebar.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))

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
        'Total_Income': applicant_income + coapplicant_income
    }

    df = pd.DataFrame(data, index=[0])

    # Feature Engineering (must match training)
    df['Log_Loan_Amount'] = np.log1p(df['LoanAmount'])
    df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 0.0001)
    df['EMI_to_Income_Ratio'] = df['EMI'] / ((df['Total_Income'] / 12) + 0.0001)

    return df

input_df = user_input_features()

# Show input summary
with st.expander("📋 View Input Summary"):
    st.dataframe(input_df, use_container_width=True)

# Preprocess input (One-hot encoding)
processed_input = pd.get_dummies(input_df)
for col in feature_names:
    if col not in processed_input.columns:
        processed_input[col] = 0
processed_input = processed_input[feature_names]


def get_shap_values_1d(explainer, processed_input, feature_names):
    """Extract 1D SHAP values for the approved class from any SHAP format."""
    shap_values = explainer.shap_values(processed_input)

    if isinstance(shap_values, list):
        # Old format: list of arrays per class
        vals = np.array(shap_values[1])
        base_val = float(np.array(explainer.expected_value).flat[1])
    else:
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            # New format: (samples, features, classes)
            vals = shap_values[:, :, 1]
            base_val = float(np.array(explainer.expected_value).flat[1])
        elif shap_values.ndim == 2:
            vals = shap_values
            base_val = float(np.array(explainer.expected_value).flat[0])
        else:
            vals = shap_values
            base_val = float(explainer.expected_value)

    # Always return a clean 1D array
    vals_1d = np.array(vals).flatten()[:len(feature_names)]
    return vals_1d, base_val


def plot_waterfall(explanation):
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    return plt.gcf()


def plot_bar_fallback(vals_1d, feature_names):
    feat_importance = pd.DataFrame({
        'Feature': list(feature_names),
        'SHAP Value': vals_1d,
        'Abs SHAP': np.abs(vals_1d)
    }).sort_values('Abs SHAP', ascending=True).tail(12)

    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in feat_importance['SHAP Value']]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feat_importance['Feature'], feat_importance['SHAP Value'], color=colors)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('SHAP Value (impact on model output)')
    ax.set_title('Feature Impact on Prediction\n🟢 Pushes toward Approval   🔴 Pushes toward Rejection')
    plt.tight_layout()
    return fig


# Predict button
if st.button("🔍 Predict Eligibility", use_container_width=True):
    prediction = model.predict(processed_input)
    probability = model.predict_proba(processed_input)
    approved = prediction[0] == 1
    confidence = max(probability[0]) * 100

    # Result display
    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Prediction Result")
        if approved:
            st.success("✅ Loan Approved!")
        else:
            st.error("❌ Loan Rejected")

        st.metric("Confidence", f"{confidence:.2f}%")
        st.metric("Approval Probability", f"{probability[0][1]*100:.2f}%")
        st.metric("Rejection Probability", f"{probability[0][0]*100:.2f}%")

    # Initialize vals_1d before the try block so it's always defined
    vals_1d = None
    base_val = None

    with col2:
        st.markdown("### 🔎 Why this decision? (SHAP Explanation)")

        try:
            vals_1d, base_val = get_shap_values_1d(explainer, processed_input, feature_names)

            explanation = shap.Explanation(
                values=vals_1d,
                base_values=base_val,
                data=processed_input.iloc[0].values,
                feature_names=list(feature_names)
            )

            try:
                fig = plot_waterfall(explanation)
                st.pyplot(fig, use_container_width=True)
                plt.clf()
            except Exception as waterfall_err:
                st.warning(f"Waterfall plot unavailable ({waterfall_err}), showing bar chart instead.")
                fig = plot_bar_fallback(vals_1d, feature_names)
                st.pyplot(fig, use_container_width=True)
                plt.clf()

        except Exception as shap_err:
            st.error(f"SHAP explanation failed: {shap_err}")

    # SHAP values table — only rendered if vals_1d was successfully computed
    st.markdown("---")
    st.markdown("### 📊 Detailed SHAP Values")

    if vals_1d is not None:
        try:
            shap_df = pd.DataFrame({
                'Feature': list(feature_names),
                'Input Value': processed_input.iloc[0].values,
                'SHAP Value': vals_1d,
                'Impact': ['🟢 Positive' if v > 0 else '🔴 Negative' for v in vals_1d]
            }).sort_values('SHAP Value', ascending=False)
            st.dataframe(shap_df, use_container_width=True)
        except Exception as table_err:
            st.warning(f"Could not render SHAP table: {table_err}")
    else:
        st.info("SHAP values unavailable — table cannot be displayed.")