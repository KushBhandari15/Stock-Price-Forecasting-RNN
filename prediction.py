import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import yfinance as yf
from datetime import datetime
from model import Forecasting_Model
from data_processing import get_ticker_data
import os

def get_prediction(ticker, window_size=10):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    try:
        scaler_X = joblib.load(os.path.join(BASE_DIR, 'X_scaler.gz'))
        scaler_y = joblib.load(os.path.join(BASE_DIR, 'y_scaler.gz'))
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(os.path.join(BASE_DIR, 'hybrid_model.json'))
        lstm_path = os.path.join(BASE_DIR, 'lstm_model.pth')
    except FileNotFoundError as e:
        print(f"Error: Missing model files at {BASE_DIR}. Have you run the training pipeline?")
        return None 

    df = get_ticker_data(ticker)
    if len(df) < 10:
        print("Not enough data points to generate a window.")
        return None
    
    recent_data = df.iloc[-window_size:].values
    n_features = recent_data.shape[1]
    scaled_data = scaler_X.transform(recent_data)
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model = Forecasting_Model(input_size=n_features, hidden_size=128, num_layers=2, output_size=1)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.to(device)
    lstm_model.eval()

    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        _, context_vector = lstm_model(input_tensor)

    context_vector = context_vector.cpu().numpy()
    prediction_scaled = xgb_model.predict(context_vector)

    prediction_real = scaler_y.inverse_transform(prediction_scaled.reshape(-1,1))
    predicted_decimal = prediction_real[0][0]
    print(f"Prediction result for {ticker}")
    print(f"Predicted Return (Decimal): {predicted_decimal:.6f}")
    print(f"Predicted Percentage Change: {predicted_decimal * 100:.2f}%")
    
    return predicted_decimal

def sanity_check():
    tickers = ['AAPL', 'NVDA', 'TSLA', 'AMD', 'MSFT']
    print(f"{'Ticker':<10} | {'Decimal':<12} | {'Percentage':<15}")
    print("-" * 45)

    for ticker in tickers:
        try:
            # We simply call your existing function
            # Note: modifying get_prediction slightly to just return the value helps, 
            # but it currently prints inside the function too, which is fine.
            pred = get_prediction(ticker)
            
            if pred is not None:
                print(f"{ticker:<10} | {pred:.6f}     | {pred * 100:.4f}%")
        except Exception as e:
            print(f"{ticker:<10} | Error: {e}")

    print("-" * 45)

if __name__ == "__main__":
    sanity_check()