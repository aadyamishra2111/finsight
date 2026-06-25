import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def train_model():
    df = pd.read_csv("data/transactions.csv")
    
    # Encode categorical columns
    le_merchant = LabelEncoder()
    le_category = LabelEncoder()
    
    df['merchant_encoded'] = le_merchant.fit_transform(df['merchant'])
    df['category_encoded'] = le_category.fit_transform(df['category'])
    
    # Features we train on
    features = ['amount', 'hour', 'day_of_week', 'merchant_encoded', 'category_encoded']
    X = df[features]
    
    # Isolation Forest — contamination = how much of data we expect to be anomalous
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # ~5% anomalies
        random_state=42
    )
    model.fit(X)
    
    # -1 means anomaly, 1 means normal — convert to 0/1
    df['predicted_anomaly'] = (model.predict(X) == -1).astype(int)
    df['anomaly_score'] = -model.score_samples(X)  # higher = more anomalous
    
    # Save model and encoders
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/encoders.pkl", "wb") as f:
        pickle.dump((le_merchant, le_category), f)
    
    # Save predictions
    df.to_csv("data/transactions_with_predictions.csv", index=False)
    
    print("Model trained and saved.")
    print(f"Predicted anomalies: {df['predicted_anomaly'].sum()}")
    print(f"Actual anomalies: {df['is_anomaly'].sum()}")
    
    return model, df

if __name__ == "__main__":
    train_model()