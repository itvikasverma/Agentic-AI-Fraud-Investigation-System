import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from ml.predict import FraudPredictor

def evaluate_ml():
    print("Evaluating ML Models...")
    if not os.path.exists('data/transactions.csv'):
        print("Data not found. Run database/seed.py first.")
        return
        
    df = pd.read_csv('data/transactions.csv')
    
    # We will use the last 20% of data for evaluation
    test_size = int(len(df) * 0.2)
    test_df = df.iloc[-test_size:]
    
    predictor = FraudPredictor()
    
    y_true = test_df['fraud_label'].tolist()
    y_pred_probs = []
    y_pred_classes = []
    
    for _, row in test_df.iterrows():
        tx_data = row.to_dict()
        res = predictor.predict(tx_data)
        
        prob = res['fraud_probability']
        y_pred_probs.append(prob)
        y_pred_classes.append(1 if prob > 0.5 else 0)
        
    precision = precision_score(y_true, y_pred_classes)
    recall = recall_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    pr_auc = average_precision_score(y_true, y_pred_probs)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("Evaluation Complete.")

if __name__ == "__main__":
    evaluate_ml()
