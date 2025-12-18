# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import DEFAULT_PERIOD, DEFAULT_INTERVAL, LOTTIE_BULL, LOTTIE_BEAR
from utils import load_stock_data, save_sklearn_model, load_sklearn_model, save_tf_model, create_report_pdf
from indicators import add_all
from models import train_linear, train_random_forest, train_lstm, train_prophet, automl_select
from sentiment import analyze_texts, get_sentiment_pipeline
from backtesting import run_backtest

st.set_page_config(page_title="Advanced Stock ML Platform", layout="wide")
# load css
if Path("styles.css").exists():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Theme toggle
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme=='light' else 'light'
    st.write(f"<script>document.documentElement.setAttribute('data-theme','{st.session_state.theme}')</script>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Controls")
source = st.sidebar.radio("Data Source", ("yfinance","upload_csv"))
period = st.sidebar.selectbox("Period", [ "1y","2y","5y","10y","max"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)
tickers_input = st.sidebar.text_input("Ticker(s) comma separated", value="AAPL")
uploaded_file = None
if source == "upload_csv":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
forecast_horizon = st.sidebar.selectbox("Forecast horizon (days)", [7,30,90], index=0)
model_selection = st.sidebar.multiselect("Models", ["RandomForest","LinearRegression","LSTM","Prophet","AutoML"], default=["RandomForest","LinearRegression"])
st.sidebar.button("Toggle Theme", on_click=toggle_theme)

# Main
st.title("Advanced Stock Market Visualization, Forecasting & Insights")

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Enter at least one ticker")
    st.stop()

# Load data for all tickers
data_store = {}
errors = {}
for t in tickers:
    df, err = load_stock_data(source, t, period=period, interval=interval, uploaded_file=uploaded_file)
    if err:
        errors[t] = err
    else:
        df = df.sort_values("date").reset_index(drop=True)
        df = add_all(df)
        data_store[t] = df

if not data_store:
    st.error("No valid data loaded.")
    for k,v in errors.items():
        st.write(f"{k}: {v}")
    st.stop()

primary = tickers[0]
df = data_store[primary]

# Top metrics
latest = df.iloc[-1]
close_val = float(latest.get('close', 0.0))
rsi_val = float(latest.get('rsi_14') or 0.0)
st.metric(f"{primary} Close", f"{close_val:.2f}")
st.metric("RSI(14)", f"{rsi_val:.2f}")

# Price chart
st.subheader(f"{primary} Price & Bollinger")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75,0.25])
fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="OHLC"), row=1, col=1)
if 'bb_upper' in df.columns:
    fig.add_trace(go.Scatter(x=df['date'], y=df['bb_upper'], name="BB Upper", line=dict(dash='dash')), row=1, col=1)
if 'bb_mid' in df.columns:
    fig.add_trace(go.Scatter(x=df['date'], y=df['bb_mid'], name="BB Mid"), row=1, col=1)
fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name="Volume"), row=2, col=1)
fig.update_layout(height=650)
st.plotly_chart(fig, use_container_width=True)

# Oscillators
st.subheader("Oscillators")
fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True)
if 'rsi_14' in df.columns:
    fig2.add_trace(go.Scatter(x=df['date'], y=df['rsi_14'], name="RSI(14)"), row=1, col=1)
if 'stoch_k' in df.columns:
    fig2.add_trace(go.Scatter(x=df['date'], y=df['stoch_k'], name="%K"), row=2, col=1)
    fig2.add_trace(go.Scatter(x=df['date'], y=df['stoch_d'], name="%D"), row=2, col=1)
fig2.update_layout(height=350)
st.plotly_chart(fig2, use_container_width=True)

# Market regime via clustering
st.subheader("Market Regime (simple clustering)")
from sklearn.cluster import KMeans
reg = df[['date']].copy()
reg['ret'] = df['close'].pct_change().fillna(0)
reg['vol'] = reg['ret'].rolling(21).std().fillna(0)
X = reg[['ret','vol']].fillna(0).values
if len(X) >= 10:
    try:
        km = KMeans(n_clusters=2, random_state=42).fit(X)
        reg['label'] = km.labels_
        means = reg.groupby('label')['ret'].mean()
        mapping = {0:'Bear',1:'Bull'} if means.iloc[0] < means.iloc[1] else {0:'Bull',1:'Bear'}
        reg['regime'] = reg['label'].map(mapping)
        st.info(f"Current regime: {reg['regime'].iloc[-1]}")
    except Exception as e:
        st.warning("Regime detection error: " + str(e))

# Sentiment
st.subheader("News Sentiment (FinBERT or VADER fallback)")
news = st.text_area("Paste headlines/articles (one per line)", height=120)
if st.button("Analyze Sentiment"):
    if not news.strip():
        st.warning("Paste text first")
    else:
        texts = [x.strip() for x in news.splitlines() if x.strip()]
        try:
            res = analyze_texts(texts)
            st.table(res)
            avg = sum([r.get('score',0) for r in res]) / max(1,len(res))
            st.metric("Avg Sentiment Score", f"{avg:.3f}")
        except Exception as e:
            st.error("Sentiment failed: " + str(e))

# Model training
st.subheader("Train Models & Forecast")
if st.button("Train Selected Models"):
    feat = df.copy().dropna().reset_index(drop=True)
    feat['target'] = feat['close'].shift(-1)
    feat = feat.dropna().reset_index(drop=True)
    features = [c for c in ['close','ema_12','ema_26','rsi_14','macd','bb_mid'] if c in feat.columns]
    if len(feat) < 20 or not features:
        st.warning("Not enough data/features to train")
    else:
        X = feat[features].fillna(method='ffill').fillna(0)
        y = feat['target']
        results = {}
        if "LinearRegression" in model_selection:
            try:
                _, m = train_linear(X,y)
                results['LinearRegression'] = m
            except Exception as e:
                results['LinearRegression_error'] = str(e)
        if "RandomForest" in model_selection:
            try:
                _, m = train_random_forest(X,y)
                results['RandomForest'] = m
            except Exception as e:
                results['RandomForest_error'] = str(e)
        if "LSTM" in model_selection:
            try:
                series = df['close'].dropna()
                model_lstm, m, _ = train_lstm(series, lookback=30, epochs=5)
                results['LSTM'] = m
            except Exception as e:
                results['LSTM_error'] = str(e)
        if "Prophet" in model_selection:
            try:
                m, forecast = train_prophet(df, periods=forecast_horizon)
                results['Prophet'] = {"forecast_rows": len(forecast)}
            except Exception as e:
                results['Prophet_error'] = str(e)
        if "AutoML" in model_selection:
            try:
                trainers = {}
                if 'RandomForest' not in results:
                    trainers['RandomForest'] = train_random_forest
                if 'LinearRegression' not in results:
                    trainers['LinearRegression'] = train_linear
                best_name, _, best_m = automl_select(trainers, X, y)
                results['AutoML'] = {"best":best_name, "metrics":best_m}
            except Exception as e:
                results['AutoML_error'] = str(e)
        st.json(results)

# Forecast & signals
st.subheader("Forecast & Buy/Sell Signals")
if st.button("Generate Forecast & Signals"):
    try:
        # try load RF
        try:
            rf = load_sklearn_model("random_forest")
        except Exception:
            rf = None
        last = df.iloc[-1]
        feat_row = {}
        for k in ['close','ema_12','ema_26','rsi_14','macd','bb_mid']:
            feat_row[k] = float(last[k]) if k in last and pd.notna(last[k]) else 0.0
        Xlast = pd.DataFrame([feat_row])
        if rf is not None:
            pred = rf.predict(Xlast)[0]
        else:
            try:
                lr = load_sklearn_model("linear_regression")
                pred = lr.predict(Xlast)[0]
            except Exception:
                pred = float(last['close'])
        st.metric("Next day predicted close", f"{float(pred):.2f}")
        signal = 1 if (pred > float(last['close']) and float(last.get('rsi_14',50)) < 70) else 0
        st.write("Signal:", "BUY" if signal==1 else "HOLD/SELL")
        signals = (df['ema_12'] > df['ema_26']).astype(int) if 'ema_12' in df.columns and 'ema_26' in df.columns else pd.Series(0, index=df.index)
        bt_df, bt_metrics = run_backtest(df, signals)
        st.write("Backtest metrics:", bt_metrics)
        st.line_chart(bt_df['portfolio'])
    except Exception as e:
        st.error("Forecast error: " + str(e))

# Multi stock compare
st.subheader("Multiple Stock Comparison")
sel = st.multiselect("Compare tickers", tickers, default=tickers[:2])
if sel:
    figc = go.Figure()
    for s in sel:
        d = data_store[s]
        figc.add_trace(go.Scatter(x=d['date'], y=d['close'], name=s))
    st.plotly_chart(figc, use_container_width=True)

# Export PDF
st.subheader("Export PDF Report")
rname = st.text_input("Report filename", value=f"{primary}_report_{datetime.now().strftime('%Y%m%d_%H%M')}")
if st.button("Generate Report"):
    lines = [
        f"Report: {primary}",
        f"Generated: {datetime.now().isoformat()}",
        f"Close: {close_val:.2f}",
        f"RSI: {rsi_val:.2f}"
    ]
    p = create_report_pdf("\n".join(lines), f"{rname}.pdf")
    if p:
        with open(p, "rb") as f:
            st.download_button("Download PDF", f, file_name=p.name, mime="application/pdf")
    else:
        st.error("Report create failed")

st.caption("Advanced Stock ML Platform — Final unified version")
