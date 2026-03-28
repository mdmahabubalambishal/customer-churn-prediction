import pickle
import pandas as pd
from src.feature_engineering import engineer_features


def load_pipeline(model_path: str = 'models/full_pipeline.pkl'):
    """Saved pipeline load"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_churn(customer_data: dict, pipeline=None) -> dict:
    """
    Single customer churn prediction

    Returns:
        churn_probability, will_churn, risk_level
    """
    if pipeline is None:
        pipeline = load_pipeline()

    df  = pd.DataFrame([customer_data])
    df  = engineer_features(df)

    prob = pipeline.predict_proba(df)[0][1]
    pred = pipeline.predict(df)[0]

    return {
        'churn_probability': round(float(prob), 4),
        'will_churn'       : bool(pred),
        'risk_level'       : (
            'HIGH'   if prob > 0.7 else
            'MEDIUM' if prob > 0.4 else
            'LOW'
        )
    }


if __name__ == "__main__":
    sample = {
        'gender'          : 'Male',
        'SeniorCitizen'   : 0,
        'Partner'         : 1,
        'Dependents'      : 0,
        'tenure'          : 2,
        'PhoneService'    : 1,
        'MultipleLines'   : 0,
        'InternetService' : 'Fiber optic',
        'OnlineSecurity'  : 0,
        'OnlineBackup'    : 0,
        'DeviceProtection': 0,
        'TechSupport'     : 0,
        'StreamingTV'     : 0,
        'StreamingMovies' : 0,
        'Contract'        : 'Month-to-month',
        'PaperlessBilling': 1,
        'PaymentMethod'   : 'Electronic check',
        'MonthlyCharges'  : 85.0,
        'TotalCharges'    : 170.0
    }
    result = predict_churn(sample)
    print(f"✅ Result: {result}")
