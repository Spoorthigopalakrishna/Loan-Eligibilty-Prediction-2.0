# How the Loan Prediction Project Works

## 🎯 High-Level Overview

This project uses **XGBoost** (Extreme Gradient Boosting) and **SHAP** (SHapley Additive exPlanations) to predict loan eligibility. An applicant's financial profile is analyzed by a highly-tuned model, which outputs an approval/rejection decision along with detailed, human-readable explanations of the factors driving that decision.

## 📊 Architecture

```
User Input (Streamlit Dashboard)
        ↓
Data Processing (Feature Engineering + Label Encoding)
        ↓
XGBoost Model (Phase 3 Optimized Classifier)
        ↓
SHAP TreeExplainer (Local & Global Interpretability)
        ↓
Visualizations (Waterfall, Force Plot, Summary Bar)
```

## 🔄 Detailed Workflow

### 1. **User Input Phase** (app.py - Sidebar)
The system collects 11 raw data points:
- **Demographics**: Gender, Marital Status, Dependents
- **Financial Info**: Applicant & Coapplicant Income, Loan Amount, Credit History
- **Employment**: Education, Self-Employed, Loan Term
- **Property**: Urban, Semiurban, or Rural location

### 2. **Data Processing Phase** (data_processing.py)

**Label Encoding**:
- Unlike Phase 1, we now use `LabelEncoder` objects fitted strictly on training data.
- This ensures that category "Male" always maps to the same integer in both training and live inference, preventing "unseen category" errors.

**Feature Engineering**:
We create 4 powerful derived features that help the model understand debt burden:
- **Total_Income**: Sum of both applicants (the true household pool).
- **Log_Loan_Amount**: Normalizes the distribution (handles outliers better).
- **EMI**: Estimated Monthly Installment (Loan / Term).
- **EMI_to_Income_Ratio**: The percentage of monthly income spent on this loan.

### 3. **Model Training Phase** (train_model.py)

**Class Imbalance Handling**:
- Lending data is naturally imbalanced (~70% approved / 30% rejected).
- We calculate `scale_pos_weight` to tell XGBoost to pay extra attention to the minority (rejected) class.
- This increased our "Rejection Recall" from **55% to 68%**.

**Hyperparameter Tuning**:
- We use `GridSearchCV` to test hundreds of combinations of `max_depth`, `learning_rate`, and `subsample`.
- The final model is optimized for **F1-Score**, ensuring a balance between accuracy and fairness.

### 4. **Explainability Phase** (SHAP Analysis)

**Global Context**:
- We generate a **Summary Plot** that ranks features by their overall impact. 
- *Insight*: Credit History and EMI-to-Income Ratio are usually the strongest predictors.

**Local Prediction (The "Why")**:
SHAP decomposes the model's output for a specific applicant:
- **Base Value**: The average likelihood of approval across all training data.
- **SHAP Values**: The "tug-of-war" between features.
- **Positive Impact (Green)**: Factors pushing toward Approval (e.g., High Income).
- **Negative Impact (Red)**: Factors pushing toward Rejection (e.g., Poor Credit).

### 5. **Visualization Phase** (Dashboard Tabs)

We provide three ways to view the results:
1.  **Waterfall Plot**: The most detailed view. Shows the journey from the base value to the final prediction.
2.  **Force Plot**: A sleek, interactive horizontal bar showing the balance of forces.
3.  **Bar Chart**: A prioritized list of the top reasons why the loan was approved or rejected.

## 🧠 Key Concepts

### Why XGBoost?
XGBoost is a state-of-the-art implementation of gradient boosted decision trees. It is significantly more powerful than Random Forest for tabular data because it iteratively corrects the mistakes of previous trees.

### Why SHAP?
Traditional "Feature Importance" only tells you what matters *on average*. SHAP tells you exactly what mattered for **this specific applicant**. It is based on game theory (Shapley Values) and is the industry gold standard for ML transparency.

## 📈 Example Prediction Flow

1.  **INPUT**: Applicant with $5k income, but $0 credit history.
2.  **MODEL**: Predicts **Rejected** (82% confidence).
3.  **SHAP**:
    - Credit_History: -0.45 (Massive negative impact)
    - EMI_Ratio: -0.10 (Negative impact)
    - Total_Income: +0.05 (Slight positive impact)
4.  **VISUALIZATION**: Waterfall plot shows the massive red bar for Credit History pulling the score down.

## 🛠 Technical Stack

- **XGBoost**: Gradient boosting framework
- **SHAP**: Interpretability
- **Streamlit**: Dashboarding
- **Joblib**: Efficient artifact serialization
- **Scikit-learn**: Preprocessing and CV-tuning

---

**Last Updated**: April 2026  
**Version**: Phase 3 — Explainability (SHAP)
