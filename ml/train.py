import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import xgboost as xgb
import joblib

def load_data():
    if not os.path.exists('data/transactions.csv'):
        raise FileNotFoundError("Transactions data not found. Run database/seed.py first.")
    df = pd.read_csv('data/transactions.csv')
    return df

def train_models():
    print("Loading data...")
    df = load_data()
    
    # Feature engineering
    features = [
        'amount', 'transaction_hour', 'weekend', 'location_change', 
        'device_change', 'international_transaction', 'failed_attempts', 
        'distance_from_previous_transaction', 'transaction_velocity', 
        'amount_deviation'
    ]
    target = 'fraud_label'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Baseline: Logistic Regression
    print("\n--- Training Logistic Regression (Baseline) ---")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    
    print("Logistic Regression Performance:")
    print(classification_report(y_test, lr_preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, lr_probs):.4f}")
    
    # 2. Main Model: XGBoost
    print("\n--- Training XGBoost Classifier ---")
    # Estimate scale_pos_weight
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    print("XGBoost Performance:")
    print(classification_report(y_test, xgb_preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, xgb_probs):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, xgb_probs):.4f}")
    
    # 3. Anomaly Detection: Isolation Forest
    print("\n--- Training Isolation Forest ---")
    # Train only on non-fraud data to learn "normal" behavior
    X_train_normal = X_train_scaled[y_train == 0]
    
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_train_normal)
    
    # For Isolation forest, -1 is anomaly, 1 is normal
    iso_preds = iso.predict(X_test_scaled)
    iso_preds = [1 if x == -1 else 0 for x in iso_preds] # Convert to 1 = fraud
    
    print("Isolation Forest Performance:")
    print(classification_report(y_test, iso_preds))
    
    # Save models
    print("\nSaving models to 'models/' directory...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(lr, 'models/logistic_regression.pkl')
    xgb_model.save_model('models/fraud_xgboost.json')
    joblib.dump(iso, 'models/isolation_forest.pkl')
    
    # Save feature names for SHAP and prediction
    joblib.dump(features, 'models/features.pkl')
    
    print("Models saved successfully!")

if __name__ == "__main__":
    train_models()
