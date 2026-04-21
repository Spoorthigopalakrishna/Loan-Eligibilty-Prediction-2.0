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
- **Polished UI (Phase 5)**: A professional, centered dashboard with card-based layouts and responsive design.

## 📁 Project Structure

```
Loan-Eligibilty-Prediction-2.0/
├── app.py                         # Professional Dashboard entry point
├── HOW_IT_WORKS.md                # Detailed technical guide
├── src/
│   ├── data_processing.py         # Feature engineering & cleaning
│   └── train_model.py             # XGBoost training & metrics generation
├── data/
│   └── raw/                       # Immutable source data (train.csv)
└── models/
    ├── loan_model.pkl             # Trained XGBoost artifact
    ├── explainer.pkl              # SHAP TreeExplainer
    ├── metrics.pkl                # Accuracy, F1, and Confusion Matrix data
    ├── confusion_matrix.png       # Test set performance visualization
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

### 2. Training (Required for Metrics)
```bash
python src/train_model.py
```
This will retrain the XGBoost model, perform GridSearchCV, and generate the `metrics.pkl` and `confusion_matrix.png` required for the dashboard.

### 3. Launch App
```bash
streamlit run app.py
```

## 📊 Performance Metrics

| Metric | Value |
|---|---|
| **Overall Accuracy** | 83.74% |
| **F1-Score (Macro)** | 0.8021 |
| **Approved (Recall)** | 91.00% |
| **Rejected (Recall)** | 68.00% |

*Tuned with: `learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.8, colsample_bytree=0.9`*

## 🔎 Interpretability Features

The dashboard provides three layers of explanation:
1. **Decision Flow (Waterfall)**: Deconstructs the probability score from base value to final prediction.
2. **Professional Status Badges**: High-visibility "ELIGIBLE" / "INELIGIBLE" indicators with confidence percentages.
3. **Impact Table**: Detailed breakdown of each feature's positive or negative contribution to the result.

## 📝 Version History

### Phase 5 — Professional Dashboard & Deployment (Current)
- ✅ Removed Sidebar for a cleaner, centered UX.
- ✅ Implemented **Responsive Card Layouts** for all devices.
- ✅ Moved technical stats to a **Model Governance Footer**.
- ✅ Pushed to GitHub and ready for **One-Click Streamlit Cloud** deployment.

### Phase 4 — Polished UI & Monitoring
- ✅ Redesigned input form for better main-panel ergonomics.
- ✅ Added **Result Badges** and **Confidence Progress Bars**.
- ✅ Automated performance artifact generation (`metrics.pkl`).

### Phase 3 — Explainability (SHAP)
- ✅ Integrated **SHAP TreeExplainer** for granular model transparency.
- ✅ Added interactive **Waterfall Plots** to UI.
- ✅ Optimized for **Class Imbalance** (increased rejection recall by 13%).

### Phase 2 — XGBoost Model
- ✅ Transitioned from RandomForest to **XGBoost**.
- ✅ Implemented **LabelEncoder** for consistent train/inference mapping.
- ✅ Integrated **GridSearchCV** for hyperparameter optimization.

## 📄 License
MIT License - See [LICENSE](LICENSE) for details.

## 👤 Author
**Spoorthi Gopalakrishna** - [GitHub Profile](https://github.com/Spoorthigopalakrishna)
