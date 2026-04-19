"""
Phase 3 -- Explainability (SHAP)
=====================================
- Uses shap.TreeExplainer for XGBoost
- Generates global summary plot
- Saves explainer and model artifacts
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# Allow running from repo root: python src/train_model.py
sys.path.insert(0, os.path.dirname(__file__))
from data_processing import get_processed_data, split_data

# ─────────────────────────────────────────────
# 1. Categorical columns that need encoding
# ─────────────────────────────────────────────
CAT_COLS = ['Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'Property_Area']


def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit LabelEncoders on X_train only, then transform both splits.
    Returns encoded DataFrames + the fitted encoder dict.
    """
    encoders = {}
    X_train = X_train.copy()
    X_test  = X_test.copy()

    for col in CAT_COLS:
        if col in X_train.columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            # Handle unseen labels in test gracefully
            X_test[col] = X_test[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
            )
            encoders[col] = le

    return X_train, X_test, encoders


# ─────────────────────────────────────────────
# 2. GridSearchCV parameter grid
# ─────────────────────────────────────────────
PARAM_GRID = {
    'n_estimators':  [100, 200, 300],
    'max_depth':     [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample':     [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}


def tune_xgboost(X_train, y_train, scale_pos_weight=1.0):
    """
    Run GridSearchCV with StratifiedKFold (handles class imbalance).
    Optimises for F1 score. Returns the best estimator.
    """
    base_model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n[INFO] Running GridSearchCV with scale_pos_weight={scale_pos_weight:.4f}...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring='f1',   # F1 handles class imbalance better than accuracy
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print(f"\n[INFO] Best params : {grid_search.best_params_}")
    print(f"[INFO] Best CV F1  : {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


# ─────────────────────────────────────────────
# 3. Main training pipeline
# ─────────────────────────────────────────────
def train():
    # 3a. Load + clean + engineer features
    print("[INFO] Loading and processing data...")
    df = get_processed_data('data/raw/train.csv')

    # 3b. Encode target label
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # 3c. Stratified 80/20 split (preserves ~70/30 class ratio)
    X_train, X_test, y_train, y_test = split_data(df, target_col='Loan_Status')

    # 3d. Label-encode categoricals (fit on train, transform test)
    print("[INFO] Encoding categorical features...")
    X_train, X_test, encoders = encode_features(X_train, X_test)

    # Ensure all columns are numeric
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test  = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

    print(f"\n[INFO] Class distribution in train : {dict(y_train.value_counts())}")
    print(f"[INFO] Features used               : {list(X_train.columns)}\n")

    # 3e. Calculate scale_pos_weight (Negative / Positive)
    # Since 1 is Approved (Majority) and 0 is Rejected (Minority),
    # scale_pos_weight should be < 1 to balance them if we consider 1 as positive.
    # XGBoost: scale_pos_weight = sum(negative instances) / sum(positive instances)
    counts = y_train.value_counts()
    neg_count = counts.get(0, 0)
    pos_count = counts.get(1, 1) # Avoid div by zero
    spw = neg_count / pos_count

    # 3f. Hyperparameter search
    model = tune_xgboost(X_train, y_train, scale_pos_weight=spw)

    # 3f. Evaluate on held-out test set
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("  TEST SET RESULTS")
    print("=" * 55)
    print(f"  Accuracy : {acc * 100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Rejected (0)', 'Approved (1)']))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 55)

    # 3g. Save all artifacts
    os.makedirs('models', exist_ok=True)

    joblib.dump(model,                    'models/loan_model.pkl')
    joblib.dump(encoders,                 'models/label_encoders.pkl')
    joblib.dump(X_train.columns.tolist(), 'models/feature_names.pkl')

    print("\n[INFO] Generating SHAP TreeExplainer and global plots...")
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, 'models/explainer.pkl')

    # Generate and save global summary plot
    shap_values = explainer.shap_values(X_train)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig('models/shap_summary_plot.png')
    plt.close()

    print("\n[DONE] All artifacts saved to models/")
    print("       -> loan_model.pkl")
    print("       -> label_encoders.pkl")
    print("       -> feature_names.pkl")
    print("       -> explainer.pkl")
    print("       -> shap_summary_plot.png")


if __name__ == '__main__':
    train()