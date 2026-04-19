"""
data_processing.py -- Phase 2
================================
Handles loading, cleaning, and feature engineering.
No PCA -- raw features kept for SHAP interpretability.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values:
      - Categorical -> mode imputation
      - Numerical   -> median imputation (robust to skew)
    """
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed',
                'Credit_History', 'Loan_Amount_Term']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    num_cols = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives four interpretable features:
      - Total_Income        : combined applicant + co-applicant income
      - Log_Loan_Amount     : log-transform to reduce right-skew
      - EMI                 : approximate monthly instalment
      - EMI_to_Income_Ratio : affordability ratio
    """
    df['Total_Income']        = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Log_Loan_Amount']     = np.log1p(df['LoanAmount'])
    df['EMI']                 = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1e-4)
    df['EMI_to_Income_Ratio'] = df['EMI'] / ((df['Total_Income'] / 12) + 1e-4)

    if 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)

    return df


def get_processed_data(file_path: str = 'data/raw/train.csv') -> pd.DataFrame:
    """
    Full pipeline: Load -> Clean -> Engineer -> Save processed CSV.
    Returns the processed DataFrame (Loan_Status column included).
    """
    df = pd.read_csv(file_path)
    df = clean_data(df)
    df = feature_engineering(df)

    df.to_csv('data/processed/processed_train.csv', index=False)
    print("[INFO] Processed dataset saved -> data/processed/processed_train.csv")
    return df


def split_data(df: pd.DataFrame, target_col: str = 'Loan_Status'):
    """
    Stratified 80/20 split -- preserves class ratio for imbalanced data (~70/30).
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Split -> Train: {len(X_train)}, Test: {len(X_test)} (Stratified)")
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    data = get_processed_data()
    X_train, X_test, y_train, y_test = split_data(data)
