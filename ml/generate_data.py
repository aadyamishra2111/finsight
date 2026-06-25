import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker('en_IN')
np.random.seed(42)
random.seed(42)

UPI_MERCHANTS = [
    "Swiggy", "Zomato", "BigBasket", "Blinkit", "Amazon",
    "Flipkart", "Myntra", "Ola", "Uber", "PhonePe",
    "Paytm", "Netflix", "Spotify", "BookMyShow", "MakeMyTrip",
    "IRCTC", "Zepto", "Dunzo", "Nykaa", "DMart"
]

CATEGORIES = {
    "Swiggy": "Food", "Zomato": "Food", "BigBasket": "Groceries",
    "Blinkit": "Groceries", "Zepto": "Groceries", "Dunzo": "Groceries",
    "Amazon": "Shopping", "Flipkart": "Shopping", "Myntra": "Shopping", "Nykaa": "Shopping",
    "Ola": "Transport", "Uber": "Transport",
    "Netflix": "Entertainment", "Spotify": "Entertainment", "BookMyShow": "Entertainment",
    "MakeMyTrip": "Travel", "IRCTC": "Travel",
    "PhonePe": "Transfer", "Paytm": "Transfer", "DMart": "Groceries"
}

def generate_transactions(n_normal=1000, n_anomalous=50):
    transactions = []
    start_date = datetime.now() - timedelta(days=90)

    # Normal transactions
    for i in range(n_normal):
        merchant = random.choice(UPI_MERCHANTS)
        category = CATEGORIES[merchant]

        # Realistic amount ranges per category
        if category == "Food":
            amount = round(random.uniform(80, 800), 2)
        elif category == "Groceries":
            amount = round(random.uniform(200, 3000), 2)
        elif category == "Shopping":
            amount = round(random.uniform(300, 5000), 2)
        elif category == "Transport":
            amount = round(random.uniform(50, 500), 2)
        elif category == "Entertainment":
            amount = round(random.uniform(99, 1500), 2)
        elif category == "Travel":
            amount = round(random.uniform(500, 15000), 2)
        else:
            amount = round(random.uniform(100, 10000), 2)

        txn_date = start_date + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(8, 23),
            minutes=random.randint(0, 59)
        )

        transactions.append({
            "txn_id": f"TXN{i:05d}",
            "user_id": f"USER{random.randint(1, 50):03d}",
            "merchant": merchant,
            "category": category,
            "amount": amount,
            "hour": txn_date.hour,
            "day_of_week": txn_date.weekday(),
            "timestamp": txn_date,
            "is_anomaly": 0
        })

    # Anomalous transactions
    for i in range(n_anomalous):
        merchant = random.choice(UPI_MERCHANTS)
        category = CATEGORIES[merchant]
        anomaly_type = random.choice(["high_amount", "odd_hour", "both"])

        if anomaly_type == "high_amount":
            amount = round(random.uniform(50000, 200000), 2)
            hour = random.randint(8, 23)
        elif anomaly_type == "odd_hour":
            amount = round(random.uniform(80, 800), 2)
            hour = random.randint(1, 5)
        else:
            amount = round(random.uniform(50000, 200000), 2)
            hour = random.randint(1, 5)

        txn_date = start_date + timedelta(
            days=random.randint(0, 90),
            hours=hour,
            minutes=random.randint(0, 59)
        )

        transactions.append({
            "txn_id": f"ATXN{i:05d}",
            "user_id": f"USER{random.randint(1, 50):03d}",
            "merchant": merchant,
            "category": category,
            "amount": amount,
            "hour": hour,
            "day_of_week": txn_date.weekday(),
            "timestamp": txn_date,
            "is_anomaly": 1
        })

    df = pd.DataFrame(transactions)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    return df

if __name__ == "__main__":
    df = generate_transactions()
    df.to_csv("data/transactions.csv", index=False)
    print(f"Generated {len(df)} transactions")
    print(f"Anomalies: {df['is_anomaly'].sum()}")
    print(df.head())