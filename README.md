# 🏦 Loan Eligibility Prediction 2.0

A professional-grade machine learning application designed to predict loan eligibility with high accuracy and full transparency. This project utilizes **XGBoost** for predictive power and **SHAP** for explainable AI (XAI) insights.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-green.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

## 🎯 Project Highlights

- **Predictive Power**: XGBoost classifier tuned via GridSearchCV (83.7% Accuracy).
- **Transparency**: Every prediction is accompanied by a SHAP explanation (Waterfall & Force plots).
- **Imbalance-Aware**: Optimized for **F1-Score** using `scale_pos_weight` to better identify high-risk rejections.
- **Robust Pipeline**: Modular source code with automated feature engineering and consistent LabelEncoding.
- **Interactive Dashboard**: Real-time inference with a feature-rich Streamlit UI.

## 📁 Project Structure

```
Loan-Eligibilty-Prediction-2.0/
├── app.py                         # Streamlit entry point
├── HOW_IT_WORKS.md                # Detailed technical guide
├── src/
│   ├── data_processing.py         # Feature engineering & cleaning
│   └── train_model.py             # XGBoost training & SHAP generation
├── data/
│   ├── raw/                       # Immutable source data
│   └── processed/                 # Engineered feature storage
└── models/
    ├── loan_model.pkl             # Trained XGBoost artifact
    ├── explainer.pkl              # SHAP TreeExplainer
    ├── label_encoders.pkl         # Fitted category encoders
    └── shap_summary_plot.png      # Global importance visualization
```

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Spoorthigopalakrishna/Loan-Eligibilty-Prediction-2.0.git
cd Loan-Eligibilty-Prediction-2.0
python -m venv venv
# Activate: .\venv\Scripts\activate (Win) or source venv/bin/activate (Mac)
pip install -r requirements.txt
```

### 2. Training (Optional)
```bash
python src/train_model.py
```
This will retrain the XGBoost model, perform GridSearchCV, and regenerate the SHAP explainer and global summary plots.

### 3. Launch App
```bash
streamlit run app.py
```

## 📊 Performance Metrics (Phase 3)

| Metric | Value |
|---|---|
| **Overall Accuracy** | 83.74% |
| **Approved (Recall)** | 91.00% |
| **Rejected (Recall)** | 68.00% |
| **Scoring Target** | F1-Score (Macro) |

*Tuned with: `learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.8, colsample_bytree=0.9`*

## 🔎 Interpretability Features

The dashboard provides three layers of explanation:
1. **Waterfall Plot**: Deconstructs the probability score from base value to final prediction.
2. **Interactive Force Plot**: Visualizes the tug-of-war between approval and rejection factors.
3. **Global Summary**: An expander showing overall feature importance across the entire dataset.

## 📝 Version History

### Phase 3 — Explainability (Current)
- ✅ Integrated **SHAP TreeExplainer** for granular model transparency.
- ✅ Added interactive **Force Plots** and **Waterfall Plots** to UI.
- ✅ Optimized for **Class Imbalance** (increased rejection recall by 13%).
- ✅ Automated Global Summary plot generation.

### Phase 2 — XGBoost Model
- ✅ Transitioned from RandomForest to **XGBoost**.
- ✅ Implemented **LabelEncoder** for consistent train/inference mapping.
- ✅ Integrated **GridSearchCV** for hyperparameter optimization.

### Phase 1 — Data Pipeline
- ✅ Established modular `src/` architecture.
- ✅ Built robust feature engineering (Total Income, EMI, Log Loan Amount).

## 📄 License
MIT License - See [LICENSE](LICENSE) for details.

## 👤 Author
**Spoorthi Gopalakrishna** - [GitHub Profile](https://github.com/Spoorthigopalakrishna)
