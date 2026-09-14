from langchain_core.tools import tool
from ml.predict import FraudPredictor
from ml.explain import SHAPExplainer
import json

try:
    predictor = FraudPredictor()
    explainer = SHAPExplainer()
except Exception as e:
    print(f"Warning: ML models not loaded. {e}")
    predictor = None
    explainer = None

@tool
def get_ml_fraud_assessment(transaction_data_json: str) -> str:
    """
    Evaluates a transaction using the ML model (XGBoost) and Anomaly Detection (Isolation Forest).
    Also provides SHAP explanations for the prediction.
    Input MUST be a valid JSON string containing transaction features.
    Required features: amount, transaction_hour, weekend, location_change, device_change, international_transaction, failed_attempts, distance_from_previous_transaction, transaction_velocity, amount_deviation.
    """
    if not predictor or not explainer:
        return json.dumps({"error": "ML models are not initialized. Please ensure models are trained."})
        
    try:
        data = json.loads(transaction_data_json)
        
        # Predict
        prediction = predictor.predict(data)
        
        # Explain
        explanation = explainer.explain(data)
        
        result = {
            "prediction": prediction,
            "explanation": explanation
        }
        
        return json.dumps(result, indent=2)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input."})
    except Exception as e:
        return json.dumps({"error": str(e)})
