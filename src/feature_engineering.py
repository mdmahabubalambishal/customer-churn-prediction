import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe clean and return"""
    df = df.copy()

    # no needed customerID
    df.drop('customerID', axis=1, inplace=True, errors='ignore')

    # TotalCharges numeric 
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce'
    )
    df['TotalCharges'].fillna(0, inplace=True)

    # Target encode
    if 'Churn' in df.columns:
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # Yes/No → binary
    yes_no_cols = [
        'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling', 'MultipleLines'
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = (df[col] == 'Yes').astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Business-driven features create"""
    df = df.copy()

    # Average monthly spend
    df['AvgMonthlySpend'] = df.apply(
        lambda x: x['TotalCharges'] / x['tenure']
        if x['tenure'] > 0 else x['MonthlyCharges'],
        axis=1
    )

    # Charge increase rate
    df['ChargeIncreaseRate'] = (
        df['MonthlyCharges'] - df['AvgMonthlySpend']
    )

    # Service count
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        if df[col].dtype == 'object':
            df[col] = (df[col] == 'Yes').astype(int)

    df['TotalServices']      = df[service_cols].sum(axis=1)
    df['IsLongTermCustomer'] = (df['tenure'] >= 24).astype(int)
    df['IsNewCustomer']      = (df['tenure'] <= 6).astype(int)
    df['IsHighValue']        = (df['MonthlyCharges'] >= 70).astype(int)

    contract_risk = {
        'Month-to-month': 3,
        'One year'      : 2,
        'Two year'      : 1
    }
    df['ContractRiskScore'] = df['Contract'].map(contract_risk)

    return df