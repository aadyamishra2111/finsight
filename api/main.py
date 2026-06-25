from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append(".")
from ml.explain import explain_transaction

app = FastAPI(title="FinSight API", description="Real-time transaction anomaly detection with ML explainability")

# Load model and data on startup
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/encoders.pkl", "rb") as f:
    le_merchant, le_category = pickle.load(f)

class Transaction(BaseModel):
    merchant: str
    amount: float
    hour: int
    day_of_week: int
    category: str

@app.get("/")
def root():
    return {"message": "FinSight is running", "status": "ok"}

@app.get("/transactions")
def get_all_transactions():
    df = pd.read_csv("data/transactions_with_predictions.csv")
    return {
        "total": len(df),
        "anomalies_detected": int(df['predicted_anomaly'].sum()),
        "transactions": df.tail(20).to_dict(orient="records")
    }

@app.get("/anomalies")
def get_anomalies():
    df = pd.read_csv("data/transactions_with_predictions.csv")
    flagged = df[df['predicted_anomaly'] == 1].sort_values('anomaly_score', ascending=False)
    return {
        "count": len(flagged),
        "transactions": flagged.head(20).to_dict(orient="records")
    }

@app.get("/explain/{txn_index}")
def explain(txn_index: int):
    try:
        result = explain_transaction(txn_index)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(txn: Transaction):
    try:
        # Encode inputs
        merchant_enc = le_merchant.transform([txn.merchant])[0] if txn.merchant in le_merchant.classes_ else 0
        category_enc = le_category.transform([txn.category])[0] if txn.category in le_category.classes_ else 0

        features = np.array([[txn.amount, txn.hour, txn.day_of_week, merchant_enc, category_enc]])
        prediction = model.predict(features)[0]
        score = -model.score_samples(features)[0]

        is_anomaly = prediction == -1

        reason = None
        if is_anomaly:
            if txn.amount > 20000:
                reason = f"Amount ₹{txn.amount} is unusually high"
            elif txn.hour < 6:
                reason = f"Transaction at {txn.hour}:00 hrs is outside normal hours"
            else:
                reason = "Unusual transaction pattern detected"

        return {
            "merchant": txn.merchant,
            "amount": float(txn.amount),
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": round(float(score), 4),
            "reason": reason if is_anomaly else "Transaction appears normal"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))