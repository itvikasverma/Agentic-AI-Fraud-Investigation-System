import shap
import joblib
import xgboost as xgb
import pandas as pd

class SHAPExplainer:
    def __init__(self, model_dir='models'):
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(f'{model_dir}/fraud_xgboost.json')
        self.features = joblib.load(f'{model_dir}/features.pkl')
        self.explainer = shap.TreeExplainer(self.xgb_model)
        
    def explain(self, transaction_data: dict, top_k=5):
        df = pd.DataFrame([transaction_data])[self.features]
        shap_values = self.explainer.shap_values(df)
        
        # Get feature importance for this specific instance
        instance_shap = shap_values[0]
        
        # Create list of (feature_name, shap_value)
        feature_contributions = []
        for i, feature in enumerate(self.features):
            feature_contributions.append({
                "feature": feature,
                "contribution": float(instance_shap[i]),
                "value": str(df.iloc[0][feature])
            })
            
        # Sort by absolute contribution to find the most impactful features
        feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return {
            "top_features": feature_contributions[:top_k]
        }
