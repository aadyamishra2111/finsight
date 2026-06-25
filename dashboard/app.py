import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Auto-generate data if not present
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

if not os.path.exists(os.path.join(base_path, "transactions_with_predictions.csv")):
    from ml.generate_data import generate_transactions
    from ml.train_model import train_model
    os.makedirs(base_path, exist_ok=True)
    df_raw = generate_transactions()
    df_raw.to_csv(os.path.join(base_path, "transactions.csv"), index=False)
    train_model()

def load_data():
    df = pd.read_csv(os.path.join(base_path, "transactions_with_predictions.csv"))
    return df

def predict_local(merchant, amount, hour, day_of_week, category):
    import pickle
    with open(os.path.join(model_path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_path, "encoders.pkl"), "rb") as f:
        le_merchant, le_category = pickle.load(f)
    merchant_enc = le_merchant.transform([merchant])[0] if merchant in le_merchant.classes_ else 0
    category_enc = le_category.transform([category])[0] if category in le_category.classes_ else 0
    features = np.array([[amount, hour, day_of_week, merchant_enc, category_enc]])
    prediction = model.predict(features)[0]
    score = -model.score_samples(features)[0]
    is_anomaly = prediction == -1
    if is_anomaly:
        if amount > 20000:
            reason = f"Amount ₹{amount} is unusually high"
        elif hour < 6:
            reason = f"Transaction at {hour}:00 hrs is outside normal hours"
        else:
            reason = "Unusual transaction pattern detected"
    else:
        reason = "Transaction appears normal"
    return {"is_anomaly": bool(is_anomaly), "anomaly_score": round(float(score), 4), "reason": reason}

def explain_local(txn_index):
    import pickle
    import shap
    df = load_data()
    with open(os.path.join(model_path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_path, "encoders.pkl"), "rb") as f:
        le_merchant, le_category = pickle.load(f)
    features = ['amount', 'hour', 'day_of_week', 'merchant_encoded', 'category_encoded']
    X = df[features]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    txn = df.iloc[txn_index]
    txn_shap = shap_values[txn_index]
    explanation = {}
    for feat, val, shap_val in zip(features, X.iloc[txn_index], txn_shap):
        explanation[feat] = {"value": round(float(val), 2), "shap_impact": round(float(shap_val), 4)}
    top_feature = max(explanation, key=lambda k: abs(explanation[k]['shap_impact']))
    reasons = {
        "amount": f"Amount ₹{txn['amount']} is unusually high",
        "hour": f"Transaction at {int(txn['hour'])}:00 hrs is outside normal hours",
        "day_of_week": "Transaction on unusual day",
        "merchant_encoded": f"Unusual activity at {txn['merchant']}",
        "category_encoded": f"Unusual spending in {txn['category']}"
    }
    return {
        "txn_id": txn['txn_id'],
        "amount": txn['amount'],
        "merchant": txn['merchant'],
        "is_flagged": bool(txn['predicted_anomaly']),
        "anomaly_score": round(float(txn['anomaly_score']), 4),
        "top_reason": reasons.get(top_feature, "Unusual pattern"),
        "shap_breakdown": explanation
    }

st.set_page_config(page_title="FinSight", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .metric-card { background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 20px; border-radius: 12px; color: white; text-align: center; }
    .alert-critical { background-color: #ff4444; padding: 10px; border-radius: 8px; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 FinSight — Transaction Anomaly Detector")
st.markdown("*Real-time UPI transaction monitoring with ML explainability*")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🚨 Anomalies", "⚡ Live Predict", "👤 User Risk", "📈 Trends"])

if page == "📊 Dashboard":
    st.header("📊 Transaction Overview")
    try:
        df = load_data()
        total = len(df)
        anomalies = int(df['predicted_anomaly'].sum())
        critical = int(df[df['anomaly_score'] > 0.7]['predicted_anomaly'].sum())
        rate = round(anomalies / total * 100, 1)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{total:,}")
        col2.metric("Anomalies Detected", anomalies, delta=f"{rate}% of total", delta_color="inverse")
        col3.metric("Critical Alerts", critical, delta="score > 0.7", delta_color="inverse")
        col4.metric("Clean Transactions", total - anomalies)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            cat_df = df.groupby('category')['amount'].sum().reset_index().sort_values('amount', ascending=False)
            fig = px.bar(cat_df, x='category', y='amount', color='category', title="Total Spend per Category")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(df, x='amount', color='predicted_anomaly', nbins=60,
                               title="Amount Distribution", color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Category Risk Scorecards")
        cat_risk = df.groupby('category').agg(total=('txn_id','count'), anomalies=('predicted_anomaly','sum'), avg_score=('anomaly_score','mean')).reset_index()
        cat_risk['risk_pct'] = (cat_risk['anomalies'] / cat_risk['total'] * 100).round(1)
        cat_risk = cat_risk.sort_values('risk_pct', ascending=False)
        cols = st.columns(len(cat_risk))
        for idx, (_, row) in enumerate(cat_risk.iterrows()):
            if idx < len(cols):
                color = "🔴" if row['risk_pct'] > 10 else "🟡" if row['risk_pct'] > 5 else "🟢"
                cols[idx].metric(f"{color} {row['category']}", f"{row['risk_pct']}% risk", f"{int(row['anomalies'])} flagged")
    except Exception as e:
        st.error(f"Error: {e}")

elif page == "🚨 Anomalies":
    st.header("🚨 Flagged Transactions")
    try:
        df = load_data()
        flagged = df[df['predicted_anomaly'] == 1].sort_values('anomaly_score', ascending=False)
        critical = flagged[flagged['anomaly_score'] > 0.7]
        if len(critical) > 0:
            st.markdown(f'<div class="alert-critical">🔴 CRITICAL: {len(critical)} transactions with anomaly score > 0.7</div>', unsafe_allow_html=True)
            st.write("")
        st.warning(f"⚠️ {len(flagged)} total anomalous transactions detected")
        csv = flagged.to_csv(index=False)
        st.download_button("📥 Download Flagged Transactions CSV", data=csv, file_name="finsight_anomalies.csv", mime="text/csv")
        st.dataframe(flagged[['txn_id','user_id','merchant','category','amount','hour','anomaly_score']].style.background_gradient(subset=['anomaly_score'], cmap='Reds'), use_container_width=True)

        st.subheader("🔎 SHAP Explanation")
        txn_index = st.number_input("Enter transaction index", min_value=0, max_value=len(df)-1, value=int(flagged.index[0]))
        if st.button("Explain with SHAP"):
            exp = explain_local(txn_index)
            col1, col2 = st.columns(2)
            with col1:
                st.json(exp)
            with col2:
                if exp.get('is_flagged'):
                    st.error(f"🚩 {exp['top_reason']}")
                else:
                    st.success("✅ Transaction appears normal")
                if 'shap_breakdown' in exp:
                    shap_data = exp['shap_breakdown']
                    features = list(shap_data.keys())
                    impacts = [shap_data[f]['shap_impact'] for f in features]
                    values = [shap_data[f]['value'] for f in features]
                    colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in impacts]
                    fig = go.Figure(go.Bar(x=impacts, y=[f"{f}={v}" for f, v in zip(features, values)], orientation='h', marker_color=colors))
                    fig.update_layout(title="SHAP Feature Impact", xaxis_title="SHAP Value", height=350)
                    st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

elif page == "⚡ Live Predict":
    st.header("⚡ Live Transaction Prediction")
    col1, col2 = st.columns(2)
    with col1:
        merchant = st.selectbox("Merchant", ["Swiggy","Zomato","BigBasket","Blinkit","Amazon","Flipkart","Myntra","Ola","Uber","PhonePe","Paytm","Netflix","Spotify","BookMyShow","MakeMyTrip","IRCTC","Zepto","Dunzo","Nykaa","DMart"])
        amount = st.number_input("Amount (₹)", min_value=1.0, max_value=500000.0, value=500.0)
    with col2:
        hour = st.slider("Hour of Transaction", 0, 23, 14)
        day = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        day_num = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day)
    category_map = {"Swiggy":"Food","Zomato":"Food","BigBasket":"Groceries","Blinkit":"Groceries","Zepto":"Groceries","Dunzo":"Groceries","Amazon":"Shopping","Flipkart":"Shopping","Myntra":"Shopping","Nykaa":"Shopping","Ola":"Transport","Uber":"Transport","Netflix":"Entertainment","Spotify":"Entertainment","BookMyShow":"Entertainment","MakeMyTrip":"Travel","IRCTC":"Travel","PhonePe":"Transfer","Paytm":"Transfer","DMart":"Groceries"}
    category = category_map[merchant]
    st.info(f"Category auto-detected: **{category}**")

    st.markdown("---")
    st.subheader("🚦 Velocity Check")
    check_user = st.text_input("User ID to check velocity", value="USER001")
    if st.button("Check Velocity"):
        try:
            df = load_data()
            user_txns = df[df['user_id'] == check_user].copy()
            user_txns['timestamp'] = pd.to_datetime(user_txns['timestamp'])
            user_txns = user_txns.sort_values('timestamp')
            flagged_velocity = False
            for i in range(len(user_txns)):
                window = user_txns.iloc[i:i+5]
                if len(window) == 5:
                    time_diff = (window['timestamp'].iloc[-1] - window['timestamp'].iloc[0]).seconds / 60
                    if time_diff <= 10:
                        flagged_velocity = True
                        break
            if flagged_velocity:
                st.error(f"🚨 VELOCITY ALERT: {check_user} made 5+ transactions within 10 minutes!")
            else:
                st.success(f"✅ {check_user} velocity looks normal")
            user_anomalies = df[(df['user_id'] == check_user) & (df['predicted_anomaly'] == 1)]
            st.metric("Total transactions", len(user_txns))
            st.metric("Flagged transactions", len(user_anomalies))
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    if st.button("🔍 Analyse Transaction"):
        result = predict_local(merchant, amount, hour, day_num, category)
        if result['is_anomaly']:
            st.error("🚩 ANOMALY DETECTED")
            st.error(f"Reason: {result['reason']}")
            st.metric("Anomaly Score", result['anomaly_score'])
            if result['anomaly_score'] > 0.7:
                st.markdown('<div class="alert-critical">🔴 CRITICAL ALERT — Immediate review recommended</div>', unsafe_allow_html=True)
        else:
            st.success("✅ Transaction looks normal")
            st.metric("Anomaly Score", result['anomaly_score'])

elif page == "👤 User Risk":
    st.header("👤 User Risk Analysis")
    try:
        df = load_data()
        user_risk = df.groupby('user_id').agg(total_txns=('txn_id','count'), total_anomalies=('predicted_anomaly','sum'), avg_amount=('amount','mean'), max_amount=('amount','max'), avg_score=('anomaly_score','mean')).reset_index()
        user_risk['risk_pct'] = (user_risk['total_anomalies'] / user_risk['total_txns'] * 100).round(1)
        user_risk = user_risk.sort_values('risk_pct', ascending=False)
        st.subheader("🏆 Highest Risk Users")
        st.dataframe(user_risk.head(20).style.background_gradient(subset=['risk_pct'], cmap='Reds'), use_container_width=True)
        st.subheader("User Anomaly Heatmap")
        top_users = user_risk.head(15)['user_id'].tolist()
        heatmap_data = df[df['user_id'].isin(top_users)].groupby(['user_id','category'])['predicted_anomaly'].sum().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='user_id', columns='category', values='predicted_anomaly').fillna(0)
        fig = px.imshow(heatmap_pivot, title="Anomalies per User per Category", color_continuous_scale='Reds', aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

elif page == "📈 Trends":
    st.header("📈 Anomaly Trends Over Time")
    try:
        df = load_data()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        daily = df.groupby('date').agg(total=('txn_id','count'), anomalies=('predicted_anomaly','sum')).reset_index()
        daily['anomaly_rate'] = (daily['anomalies'] / daily['total'] * 100).round(1)
        fig = px.line(daily, x='date', y='anomalies', title="Anomalies Detected Per Day", markers=True, color_discrete_sequence=['#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=daily['date'], y=daily['total'], name='Total', marker_color='#3498db'))
        fig2.add_trace(go.Bar(x=daily['date'], y=daily['anomalies'], name='Anomalies', marker_color='#e74c3c'))
        fig2.update_layout(barmode='overlay', title="Volume vs Anomalies Over Time")
        st.plotly_chart(fig2, use_container_width=True)
        hourly = df.groupby('hour').agg(total=('txn_id','count'), anomalies=('predicted_anomaly','sum')).reset_index()
        hourly['rate'] = (hourly['anomalies']/hourly['total']*100).round(1)
        fig3 = px.bar(hourly, x='hour', y='rate', title="Anomaly Rate by Hour", color='rate', color_continuous_scale='Reds')
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")