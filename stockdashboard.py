import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

st.set_page_config(page_title="Advanced Stock Dashboard", layout="wide")

st.title("📊 Stock Price Analysis and Prediction")

ticker = st.sidebar.text_input("Enter Stock Ticker", "TSLA")
period = st.sidebar.selectbox("Select Period", ["6mo", "1y", "5y"])


stock = yf.Ticker(ticker)
data = stock.history(period="max")
data.reset_index(inplace=True)

if not data.empty:

    #  1. STOCK PRICE TREND
    
    st.subheader("📈 Stock Price Trend")

    fig1 = px.line(data, x="Date", y="Close", title="Closing Price Trend")
    st.plotly_chart(fig1, use_container_width=True)

    
    #  2. MOVING AVERAGES
    
    st.subheader("📉 Moving Averages (20 & 50 Days)")

    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()

    fig2 = px.line(
        data,
        x="Date",
        y=["Close", "MA20", "MA50"],
        title="Moving Averages"
    )
    st.plotly_chart(fig2, use_container_width=True)

   
    # 3. RETURNS & VOLATILITY
   
    st.subheader("📊 Returns & Volatility Analysis")

    data["Daily Return"] = data["Close"].pct_change()
    data["Volatility"] = data["Daily Return"].rolling(20).std()

    fig3 = px.line(
        data,
        x="Date",
        y="Volatility",
        title="20-Day Rolling Volatility"
    )
    st.plotly_chart(fig3, use_container_width=True)

    data["Cumulative Return"] = (1 + data["Daily Return"]).cumprod()

    fig3b = px.line(
        data,
        x="Date",
        y="Cumulative Return",
        title="Cumulative Returns Over Time"
    )
    st.plotly_chart(fig3b, use_container_width=True)

    
    #  4. ACTUAL vs PREDICTED PRICE
    
    st.subheader("🤖 Actual vs Predicted Price (Polynomial + Ridge)")

    df = data.copy()
    df["Days"] = np.arange(len(df))

    X = df[["Days"]]
    y = df["Close"]

    
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_poly, y)

    df["Predicted"] = model.predict(X_poly)

    fig4 = px.line(
        df,
        x="Date",
        y=["Close", "Predicted"],
        title="Actual vs Polynomial Ridge Prediction"
    )

    st.plotly_chart(fig4, use_container_width=True)

else:
    st.error("No data found for this ticker.")