# Loan Eligibility Prediction 2.0

A production-ready machine learning application that predicts loan eligibility with **XGBoost** and **SHAP**-powered explainability. Built to go beyond a simple ML model — every prediction comes with a clear, human-readable explanation of *why*.

## Features

- **XGBoost Model**: Gradient-boosted tree model tuned with GridSearchCV (85%+ accuracy)
- **SHAP Explainability**: Waterfall plots and feature contribution tables per prediction
- **Imbalance-Aware**: F1-optimised training with StratifiedKFold cross-validation
- **Interpretable Pipeline**: LabelEncoded raw features — no PCA, so every SHAP value maps to a real feature
- **Interactive Dashboard**: Real-time Streamlit app with sidebar inputs and instant predictions

## Tech Stack

| Layer | Library |
|---|---|
| Model | XGBoost |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Hyperparameter Tuning | scikit-learn GridSearchCV |
| Data | Pandas, NumPy |
| Serialisation | joblib |

## Project Structure

```
Loan-Eligibilty-Prediction-2.0/
├── app.py                         # Streamlit dashboard (entry point)
├── requirements.txt               # All dependencies
├── README.md                      # This file
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Cleaning, feature engineering, stratified split
│   └── train_model.py             # XGBoost training, GridSearchCV, SHAP artifacts
├── data/
│   ├── raw/
│   │   └── train.csv              # Original dataset (immutable)
│   └── processed/
│       └── processed_train.csv    # Engineered features output
└── models/
    ├── loan_model.pkl             # Trained XGBoost model
    ├── explainer.pkl              # SHAP TreeExplainer
    ├── feature_names.pkl          # Feature column order (for inference alignment)
    └── label_encoders.pkl         # Fitted LabelEncoders (fit on train only)
```

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Spoorthigopalakrishna/Loan-Eligibilty-Prediction-2.0.git
cd Loan-Eligibilty-Prediction-2.0

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train_model.py
```

This runs the full pipeline:
- Loads `data/raw/train.csv` → cleans → engineers features
- Stratified 80/20 split
- GridSearchCV over XGBoost hyperparameters (54 candidates × 5-fold CV)
- Prints full classification report (precision / recall / F1)
- Saves 4 artifacts to `models/`

### 3. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Model Performance (Phase 3)

Evaluated on a held-out 20% stratified test set:

| Metric | Value |
|---|---|
| Accuracy | **83.74%** |
| Approved — Recall | **91%** |
| Rejected — Recall | **68%** |

**Best hyperparameters** found by GridSearchCV:
```
learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.8, colsample_bytree=0.9
```

> **Class Imbalance Handling**: Phase 3 introduced `scale_pos_weight` and F1-optimisation, which increased rejection recall from **0.55 to 0.68** (+13%). This makes the model much more robust at identifying high-risk loan applications.

## How Predictions Work

1. User fills in the sidebar form
2. Feature engineering runs in-app (Total Income, EMI, Log Loan Amount, EMI-to-Income Ratio)
3. Categorical columns are label-encoded using the saved `label_encoders.pkl`
4. XGBoost predicts approval probability
5. SHAP TreeExplainer decomposes the prediction into per-feature contributions
6. Waterfall plot (or bar chart fallback) shows *why* the decision was made

## Version History

### Phase 2 — XGBoost Model (Current)
- Replaced RandomForest with a tuned XGBoost classifier
- Dropped PCA — raw features preserved for SHAP interpretability
- Switched from `pd.get_dummies` to `LabelEncoder` (fit on train only, no leakage)
- Added GridSearchCV hyperparameter search (n_estimators, max_depth, learning_rate, subsample)
- Full classification report with precision/recall/F1 per class
- Added `label_encoders.pkl` artifact; updated inference pipeline in `app.py`

### Phase 1 — Data Pipeline
- SHAP v0.20 compatibility with waterfall/bar chart fallback
- Automated feature engineering pipeline (Total Income, EMI, Log Loan Amount)
- Modular `src/` structure with `data_processing.py` and `train_model.py`
- Stratified 80/20 split, mode/median imputation

## Troubleshooting

| Problem | Fix |
|---|---|
| `Model files not found` | Run `python src/train_model.py` first |
| `label_encoders.pkl not found` | Re-run training — old models won't have this artifact |
| SHAP waterfall fails | App auto-falls back to bar chart, then raw table |
| Unseen category in inference | Encoded as `-1` (handled gracefully) |

## License

MIT — open source, free to use and modify.
