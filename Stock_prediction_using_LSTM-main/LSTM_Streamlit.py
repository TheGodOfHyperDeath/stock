import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.metrics import MeanSquaredError
import io
import matplotlib.pyplot as plt
import yfinance as yf

# Load stock data
def load_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Check for multi-level columns and flatten them if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(1)  # Flatten the multi-level column index

    # Now access 'Adj Close' column correctly
    data['Return'] = data['Adj Close'].pct_change()
    data['RSI'] = compute_rsi(data['Adj Close'])
    data['EMA'] = data['Adj Close'].ewm(span=14, adjust=False).mean()
    return data


# Calculate RSI (Relative Strength Index)
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Prepare data for LSTM
def prepare_data(data, lookback=14):
    features = ['Adj Close', 'Volume', 'RSI', 'EMA']  # Ensure this matches the features used during training
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, 0])  # Target is the first column: 'Adj Close'
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, lookback)
    
    # Split data into training and testing
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler, features

# Load the model and make predictions
def load_and_predict(model_file, X_test, scaler, features):
    # Save the uploaded model file to a temporary location
    model_path = "/tmp/lstm_model.h5"
    with open(model_path, 'wb') as f:
        f.write(model_file.read())
    
    # Load the model from the temporary file
    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    
    # Print the model summary to understand its input shape
    model.summary()

    # Check the shape of X_test before prediction
    st.write("Shape of X_test before prediction:", X_test.shape)
    
    # Reshape X_test if needed: (samples, timesteps, features)
    if len(X_test.shape) == 2:  # If it is 2D, reshape to 3D
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))  # Add a dimension for timesteps (1 in this case)
    
    st.write("Shape of X_test after reshaping:", X_test.shape)

    # Make predictions
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        st.write(f"Error during prediction: {e}")
        return None

    # Rescale predictions
    def rescale(data, predictions):
        dummy_features = np.zeros((len(predictions), len(features) - 1))
        rescaled = scaler.inverse_transform(np.concatenate([predictions, dummy_features], axis=1))
        return rescaled[:, 0]
    
    y_pred_rescaled = rescale(data[features], y_pred)
    
    return y_pred_rescaled

# Streamlit UI
st.title("Stock Price Prediction Using LSTM")

# Input fields for stock symbol, start and end date
stock_symbol = st.text_input("Enter Stock Symbol (e.g. AAPL):", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
end_date = st.date_input("End Date", pd.to_datetime('2021-01-01'))

# Load data
if stock_symbol and start_date and end_date:
    data = load_stock_data(stock_symbol, start_date, end_date)
    st.write("Stock Data:", data.head())

    # Prepare data for LSTM
    X_train, X_test, y_train, y_test, scaler, features = prepare_data(data)

    # Show the model upload button
    model_file = st.file_uploader("Upload the trained LSTM model", type=["h5"])

    if model_file:
        y_pred_rescaled = load_and_predict(model_file, X_test, scaler, features)
        
        if y_pred_rescaled is not None:
            # Plot predicted vs actual stock prices
            plt.figure(figsize=(10, 6))
            plt.plot(y_test, label="Actual Prices", alpha=0.7)
            plt.plot(y_pred_rescaled, label="Predicted Prices", alpha=0.7)
            plt.title("Stock Price Prediction")
            plt.xlabel("Time")
            plt.ylabel("Stock Price")
            plt.legend()
            st.pyplot(plt)

