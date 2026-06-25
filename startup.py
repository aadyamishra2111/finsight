import os
if not os.path.exists("data/transactions_with_predictions.csv"):
    from ml.generate_data import generate_transactions
    from ml.train_model import train_model
    df = generate_transactions()
    df.to_csv("data/transactions.csv", index=False)
    train_model()