import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
import xgboost as xgb
import joblib

def evaluate_xgboost():
    print("Loading test data...")
    if not os.path.exists('data/transactions.csv'):
        raise FileNotFoundError("Run database/seed.py first.")
    
    df = pd.read_csv('data/transactions.csv')
    
    # Same features as train.py
    features = joblib.load('models/features.pkl')
    target = 'fraud_label'
    
    X = df[features]
    y = df[target]
    
    # Split to get the exact same test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Loading XGBoost model...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('models/fraud_xgboost.json')
    
    # Predict
    preds = xgb_model.predict(X_test)
    probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # Print Text Report
    print("\n" + "="*40)
    print("XGBOOST EVALUATION REPORT")
    print("="*40)
    print(classification_report(y_test, preds))
    
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # 1. Confusion Matrix Plot
    print("\nGenerating Confusion Matrix Plot...")
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Safe', 'Fraud'], 
                yticklabels=['Safe', 'Fraud'])
    plt.title('XGBoost Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve Plot
    print("Generating ROC Curve Plot...")
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve.png', bbox_inches='tight')
    plt.close()
    
    # 3. Precision-Recall Curve Plot
    print("Generating Precision-Recall Curve Plot...")
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('plots/pr_curve.png', bbox_inches='tight')
    plt.close()
    
    # 4. Feature Importance Plot
    print("Generating Feature Importance Plot...")
    importance = xgb_model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', hue='Feature', legend=False)
    plt.title('XGBoost Feature Importance')
    plt.savefig('plots/feature_importance.png', bbox_inches='tight')
    plt.close()
    
    print("\n✅ Evaluation complete! All visual plots have been saved in the 'plots/' folder.")

if __name__ == "__main__":
    evaluate_xgboost()
