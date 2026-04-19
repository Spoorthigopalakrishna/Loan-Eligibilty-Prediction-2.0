
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def clean_data(df):
    """
    Cleans the loan dataset: handles missing values with median/mode.
    """
    # Categorical columns - Mode Imputation
    # Note: Loan_Amount_Term is categorical in nature (fixed terms)
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Numerical columns - Median Imputation (better for skewed data like Income/LoanAmount)
    num_cols = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df

def feature_engineering(df):
    """
    Applies feature engineering: total income, EMI-to-income ratio, log loan amount.
    """
    # 1. Total Income
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    
    # 2. Log Loan Amount (to handle skewness)
    df['Log_Loan_Amount'] = np.log1p(df['LoanAmount'])
    
    # 3. EMI-to-Income Ratio
    # EMI Calculation (Approximate): LoanAmount / Loan_Amount_Term
    # We add a small epsilon to avoid division by zero
    df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 0.0001)
    # Monthly Income Ratio
    df['EMI_to_Income_Ratio'] = df['EMI'] / ((df['Total_Income'] / 12) + 0.0001)
    
    # Drop Loan_ID
    if 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)
        
    return df

def get_processed_data(file_path='data/raw/train.csv'):
    """
    Full pipeline: Load -> Clean -> Engineer -> Save
    """
    df = pd.read_csv(file_path)
    df = clean_data(df)
    df = feature_engineering(df)
    
    # Save processed dataset
    df.to_csv('data/processed/processed_train.csv', index=False)
    print("Processed dataset saved to data/processed/processed_train.csv")
    
    return df

def split_data(df, target_col='Loan_Status'):
    """
    80/20 Stratified Split
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Data Split: Train={len(X_train)}, Test={len(X_test)} (Stratified)")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data = get_processed_data()
    X_train, X_test, y_train, y_test = split_data(data)
