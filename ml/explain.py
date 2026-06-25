import pandas as pd
import numpy as np
import shap
import pickle

def explain_transaction(txn_index: int):
    # Load model and encoders
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/encoders.pkl", "rb") as f:
        le_merchant, le_category = pickle.load(f)
    
    df = pd.read_csv("data/transactions_with_predictions.csv")
    features = ['amount', 'hour', 'day_of_week', 'merchant_encoded', 'category_encoded']
    X = df[features]
    
    # SHAP explainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Get explanation for specific transaction
    txn = df.iloc[txn_index]
    txn_shap = shap_values[txn_index]
    
    explanation = {}
    for feat, val, shap_val in zip(features, X.iloc[txn_index], txn_shap):
        explanation[feat] = {
            "value": round(float(val), 2),
            "shap_impact": round(float(shap_val), 4)
        }
    
    # Human readable reason
    top_feature = max(explanation, key=lambda k: abs(explanation[k]['shap_impact']))
    
    reasons = {
        "amount": f"Transaction amount ₹{txn['amount']} is unusually high",
        "hour": f"Transaction at {int(txn['hour'])}:00 hrs is outside normal hours",
        "day_of_week": "Transaction happened on an unusual day",
        "merchant_encoded": f"Unusual activity at {txn['merchant']}",
        "category_encoded": f"Unusual spending in {txn['category']} category"
    }
    
    return {
        "txn_id": txn['txn_id'],
        "amount": txn['amount'],
        "merchant": txn['merchant'],
        "is_flagged": bool(txn['predicted_anomaly']),
        "anomaly_score": round(float(txn['anomaly_score']), 4),
        "top_reason": reasons.get(top_feature, "Unusual pattern detected"),
        "shap_breakdown": explanation
    }

if __name__ == "__main__":
    # Test on first flagged transaction
    df = pd.read_csv("data/transactions_with_predictions.csv")
    flagged = df[df['predicted_anomaly'] == 1].index[0]
    result = explain_transaction(flagged)
    print(f"\nTransaction: {result['txn_id']}")
    print(f"Amount: ₹{result['amount']}")
    print(f"Flagged: {result['is_flagged']}")
    print(f"Reason: {result['top_reason']}")
    print(f"Anomaly Score: {result['anomaly_score']}")