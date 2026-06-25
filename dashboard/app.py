import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="FinSight", page_icon="🔍", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .alert-critical { background-color: #ff4444; padding: 10px; border-radius: 8px; color: white; }
    .alert-warning { background-color: #ff8800; padding: 10px; border-radius: 8px; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 FinSight — Transaction Anomaly Detector")
st.markdown("*Real-time UPI transaction monitoring with ML explainability*")

# ── Sidebar ──────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🚨 Anomalies", "⚡ Live Predict", "👤 User Risk", "📈 Trends"])

# ── Helper ───────────────────────────────────────────
def load_data():
    res = requests.get(f"{API_URL}/transactions?limit=10000")
    data = res.json()
    df = pd.DataFrame(data['transactions'])
    # Load full data directly
    import pandas as pd_inner
    df_full = pd_inner.read_csv("data/transactions_with_predictions.csv")
    return df_full, data

# ── Dashboard Page ────────────────────────────────────
if page == "📊 Dashboard":
    st.header("📊 Transaction Overview")

    try:
        df, data = load_data()

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
            st.subheader("Spend by Category")
            cat_df = df.groupby('category')['amount'].sum().reset_index().sort_values('amount', ascending=False)
            fig = px.bar(cat_df, x='category', y='amount', color='category',
                        title="Total Spend per Category")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Normal vs Anomalous Amounts")
            fig2 = px.histogram(df, x='amount', color='predicted_anomaly',
                                nbins=60, title="Amount Distribution",
                                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                                labels={'predicted_anomaly': 'Is Anomaly'})
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Category Risk Scorecards")
        cat_risk = df.groupby('category').agg(
            total=('txn_id', 'count'),
            anomalies=('predicted_anomaly', 'sum'),
            avg_score=('anomaly_score', 'mean')
        ).reset_index()
        cat_risk['risk_pct'] = (cat_risk['anomalies'] / cat_risk['total'] * 100).round(1)
        cat_risk = cat_risk.sort_values('risk_pct', ascending=False)

        cols = st.columns(len(cat_risk))
        for i, row in cat_risk.iterrows():
            col_idx = list(cat_risk.index).index(i)
            if col_idx < len(cols):
                color = "🔴" if row['risk_pct'] > 10 else "🟡" if row['risk_pct'] > 5 else "🟢"
                cols[col_idx].metric(
                    f"{color} {row['category']}",
                    f"{row['risk_pct']}% risk",
                    f"{int(row['anomalies'])} flagged"
                )

    except Exception as e:
        st.error(f"Error: {e}")

# ── Anomalies Page ────────────────────────────────────
elif page == "🚨 Anomalies":
    st.header("🚨 Flagged Transactions")

    try:
        df, _ = load_data()
        flagged = df[df['predicted_anomaly'] == 1].sort_values('anomaly_score', ascending=False)

        critical = flagged[flagged['anomaly_score'] > 0.7]
        if len(critical) > 0:
            st.markdown(f'<div class="alert-critical">🔴 CRITICAL: {len(critical)} transactions with anomaly score > 0.7</div>', unsafe_allow_html=True)
            st.write("")

        st.warning(f"⚠️ {len(flagged)} total anomalous transactions detected")

        # Download button
        csv = flagged.to_csv(index=False)
        st.download_button(
            label="📥 Download Flagged Transactions CSV",
            data=csv,
            file_name="finsight_anomalies.csv",
            mime="text/csv"
        )

        st.dataframe(
            flagged[['txn_id','user_id','merchant','category','amount','hour','anomaly_score']]
            .style.background_gradient(subset=['anomaly_score'], cmap='Reds'),
            use_container_width=True
        )

        st.subheader("🔎 SHAP Explanation")
        txn_index = st.number_input("Enter transaction index", min_value=0, max_value=len(df)-1, value=int(flagged.index[0]))

        if st.button("Explain with SHAP"):
            exp = requests.get(f"{API_URL}/explain/{txn_index}").json()

            col1, col2 = st.columns(2)
            with col1:
                st.json(exp)
            with col2:
                if exp.get('is_flagged'):
                    st.error(f"🚩 {exp['top_reason']}")
                else:
                    st.success("✅ Transaction appears normal")

                # SHAP waterfall chart
                if 'shap_breakdown' in exp:
                    shap_data = exp['shap_breakdown']
                    features = list(shap_data.keys())
                    impacts = [shap_data[f]['shap_impact'] for f in features]
                    values = [shap_data[f]['value'] for f in features]

                    colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in impacts]

                    fig = go.Figure(go.Bar(
                        x=impacts,
                        y=[f"{f} = {v}" for f, v in zip(features, values)],
                        orientation='h',
                        marker_color=colors
                    ))
                    fig.update_layout(
                        title="SHAP Feature Impact (Red = pushes toward anomaly)",
                        xaxis_title="SHAP Value",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

# ── Live Predict Page ─────────────────────────────────
elif page == "⚡ Live Predict":
    st.header("⚡ Live Transaction Prediction")
    st.markdown("Simulate a new transaction and see if it gets flagged in real time")

    col1, col2 = st.columns(2)

    with col1:
        merchant = st.selectbox("Merchant", ["Swiggy","Zomato","BigBasket","Blinkit","Amazon",
                                              "Flipkart","Myntra","Ola","Uber","PhonePe",
                                              "Paytm","Netflix","Spotify","BookMyShow","MakeMyTrip",
                                              "IRCTC","Zepto","Dunzo","Nykaa","DMart"])
        amount = st.number_input("Amount (₹)", min_value=1.0, max_value=500000.0, value=500.0)

    with col2:
        hour = st.slider("Hour of Transaction", 0, 23, 14)
        day = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        day_num = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day)

    category_map = {
        "Swiggy":"Food","Zomato":"Food","BigBasket":"Groceries","Blinkit":"Groceries",
        "Zepto":"Groceries","Dunzo":"Groceries","Amazon":"Shopping","Flipkart":"Shopping",
        "Myntra":"Shopping","Nykaa":"Shopping","Ola":"Transport","Uber":"Transport",
        "Netflix":"Entertainment","Spotify":"Entertainment","BookMyShow":"Entertainment",
        "MakeMyTrip":"Travel","IRCTC":"Travel","PhonePe":"Transfer","Paytm":"Transfer","DMart":"Groceries"
    }
    category = category_map[merchant]
    st.info(f"Category auto-detected: **{category}**")

    # Velocity check warning
    st.markdown("---")
    st.subheader("🚦 Velocity Check")
    check_user = st.text_input("User ID to check velocity (e.g. USER001)", value="USER001")
    if st.button("Check Velocity"):
        try:
            df, _ = load_data()
            user_txns = df[df['user_id'] == check_user].copy()
            user_txns['timestamp'] = pd.to_datetime(user_txns['timestamp'])
            user_txns = user_txns.sort_values('timestamp')

            # Check for 5+ transactions within any 10-minute window
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

            user_anomalies = user_txns[user_txns['predicted_anomaly'] == 1]
            st.metric("Total transactions", len(user_txns))
            st.metric("Flagged transactions", len(user_anomalies))
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    if st.button("🔍 Analyse Transaction"):
        payload = {
            "merchant": merchant,
            "amount": amount,
            "hour": hour,
            "day_of_week": day_num,
            "category": category
        }
        res = requests.post(f"{API_URL}/predict", json=payload)
        result = res.json()

        if result['is_anomaly']:
            st.error("🚩 ANOMALY DETECTED")
            st.error(f"Reason: {result['reason']}")
            st.metric("Anomaly Score", result['anomaly_score'])
            if result['anomaly_score'] > 0.7:
                st.markdown('<div class="alert-critical">🔴 CRITICAL ALERT — Immediate review recommended</div>', unsafe_allow_html=True)
        else:
            st.success("✅ Transaction looks normal")
            st.metric("Anomaly Score", result['anomaly_score'])

# ── User Risk Page ────────────────────────────────────
elif page == "👤 User Risk":
    st.header("👤 User Risk Analysis")

    try:
        df, _ = load_data()

        user_risk = df.groupby('user_id').agg(
            total_txns=('txn_id', 'count'),
            total_anomalies=('predicted_anomaly', 'sum'),
            avg_amount=('amount', 'mean'),
            max_amount=('amount', 'max'),
            avg_score=('anomaly_score', 'mean')
        ).reset_index()

        user_risk['risk_pct'] = (user_risk['total_anomalies'] / user_risk['total_txns'] * 100).round(1)
        user_risk = user_risk.sort_values('risk_pct', ascending=False)

        st.subheader("🏆 Highest Risk Users")
        st.dataframe(
            user_risk.head(20).style.background_gradient(subset=['risk_pct'], cmap='Reds'),
            use_container_width=True
        )

        st.subheader("User Anomaly Heatmap")
        top_users = user_risk.head(15)['user_id'].tolist()
        heatmap_data = df[df['user_id'].isin(top_users)].groupby(['user_id', 'category'])['predicted_anomaly'].sum().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='user_id', columns='category', values='predicted_anomaly').fillna(0)

        fig = px.imshow(heatmap_pivot,
                       title="Anomalies per User per Category (Top 15 Riskiest Users)",
                       color_continuous_scale='Reds',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

# ── Trends Page ───────────────────────────────────────
elif page == "📈 Trends":
    st.header("📈 Anomaly Trends Over Time")

    try:
        df, _ = load_data()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date

        daily = df.groupby('date').agg(
            total=('txn_id', 'count'),
            anomalies=('predicted_anomaly', 'sum')
        ).reset_index()
        daily['anomaly_rate'] = (daily['anomalies'] / daily['total'] * 100).round(1)

        st.subheader("Daily Anomaly Count")
        fig = px.line(daily, x='date', y='anomalies', title="Anomalies Detected Per Day",
                     markers=True, color_discrete_sequence=['#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Transaction Volume vs Anomalies")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=daily['date'], y=daily['total'], name='Total', marker_color='#3498db'))
        fig2.add_trace(go.Bar(x=daily['date'], y=daily['anomalies'], name='Anomalies', marker_color='#e74c3c'))
        fig2.update_layout(barmode='overlay', title="Volume vs Anomalies Over Time")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Anomaly Rate % by Hour of Day")
        hourly = df.groupby('hour').agg(
            total=('txn_id','count'),
            anomalies=('predicted_anomaly','sum')
        ).reset_index()
        hourly['rate'] = (hourly['anomalies']/hourly['total']*100).round(1)
        fig3 = px.bar(hourly, x='hour', y='rate', title="Anomaly Rate by Hour",
                     color='rate', color_continuous_scale='Reds')
        st.plotly_chart(fig3, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")