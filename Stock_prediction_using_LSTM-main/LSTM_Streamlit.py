import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import io
import os

# Step 1: Load the Stock Data
def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Inspect and display the column names and a preview of the data for debugging
    st.write("Columns in the data:", data.columns)
    st.write("First few rows of the data:", data.head())

    # Check for the adjusted close column, or fall back to 'Close' if it's missing
    if 'Adj Close' in data.columns:
        st.write("Using 'Adj Close' for calculations.")
        close_column = 'Adj Close'
    elif 'Close' in data.columns:
        st.write("Using 'Close' for calculations.")
        close_column = 'Close'
    else:
        st.error(f"Neither 'Adj Close' nor 'Close' found in the data for {ticker}.")
        return pd.DataFrame()  # Return empty dataframe if neither column is found
    
    # Calculate returns and other indicators
    data['Return'] = data[close_column].pct_change()
    data['RSI'] = ta.momentum.RSIIndicator(data[close_column].squeeze()).rsi()
    data['EMA'] = ta.trend.EMAIndicator(data[close_column].squeeze()).ema_indicator()
    data.dropna(inplace=True)
    
    return data

# Step 2: Prepare the Dataset
def prepare_data(data, lookback=14):
    # Ensure 'Adj Close' is included as a feature
    features = ['Adj Close', 'Volume', 'RSI', 'EMA']  # Make sure 'Adj Close' is included as the first feature
    
    available_features = [col for col in features if col in data.columns]
    if len(available_features) == 0:
        st.error("None of the required columns are available in the data.")
        return None, None, None, None, None, None
    
    st.write("Using the following features for training:", available_features)
    
    # Scale data using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[available_features])

    # Function to create sequences for LSTM
    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])  # Select time_steps (lookback period)
            y.append(data[i, 0])  # Target is the first column: 'Adj Close' (price)
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, lookback)
    
    # Split data into training and testing sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler, available_features




# Step 4: Streamlit Web App
st.title("Stock Price Prediction with LSTM")
st.markdown("This app uses an LSTM model to predict stock prices based on historical data.")

# User Input for Stock
stock_symbol = st.text_input("Enter Stock Ticker", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

# Load data and prepare for prediction
data = load_stock_data(stock_symbol, start_date, end_date)

# If no data is found, stop further execution
if data.empty:
    st.stop()

X_train, X_test, y_train, y_test, scaler, features = prepare_data(data)

# File uploader for LSTM model
model_file = st.file_uploader("Upload your LSTM model", type=["h5"])

if model_file is not None:
    # Load and make predictions with the model
    y_pred_rescaled = load_and_predict(model_file, X_test, scaler, features)
    
    # Plot True vs Predicted Prices
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="True Prices", alpha=0.7)
    plt.plot(y_pred_rescaled, label="Predicted Prices", alpha=0.7)
    plt.title(f"True vs Predicted Prices for {stock_symbol}")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

    # Show Prediction Results
    st.write(f"Predicted Prices for {stock_symbol} (Last 5 predictions):")
    st.write(y_pred_rescaled[:5])
else:
    st.write("Please upload a model file to proceed.")
