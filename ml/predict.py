import joblib
import xgboost as xgb
import pandas as pd
import numpy as np

class FraudPredictor:
    def __init__(self, model_dir='models'):
        self.scaler = joblib.load(f'{model_dir}/scaler.pkl')
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(f'{model_dir}/fraud_xgboost.json')
        self.iso_model = joblib.load(f'{model_dir}/isolation_forest.pkl')
        self.features = joblib.load(f'{model_dir}/features.pkl')
        
    def predict(self, transaction_data: dict):
        """
        transaction_data should be a dictionary matching the features.
        """
        df = pd.DataFrame([transaction_data])[self.features]
        
        # XGBoost prediction
        fraud_prob = float(self.xgb_model.predict_proba(df)[0, 1])
        
        # Isolation Forest prediction
        df_scaled = self.scaler.transform(df)
        # ISF score: lower is more anomalous. We'll invert it for intuitive "anomaly score"
        raw_score = self.iso_model.score_samples(df_scaled)[0]
        # Normalize roughly to 0-1 (this is an approximation)
        anomaly_score = float(1.0 / (1.0 + np.exp(raw_score * 5))) 
        
        risk_level = "LOW"
        if fraud_prob > 0.90:
            risk_level = "CRITICAL"
        elif fraud_prob > 0.70:
            risk_level = "HIGH"
        elif fraud_prob > 0.30:
            risk_level = "MEDIUM"
            
        return {
            "fraud_probability": fraud_prob,
            "anomaly_score": anomaly_score,
            "risk_level": risk_level
        }
