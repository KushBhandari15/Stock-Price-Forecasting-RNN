import pandas as pd
import torch.nn as nn
from sklearn.metrics import accuracy_score
import numpy as np
import yfinance as yf

def bollinger_bands(data, window=20, num_std=2):
    """
    bb_value[t] = (price[t] - SMA[t])/(2 * stdev[t])

    bb_value = 0 → price is at the SMA
    bb_value = +1 → price is at the upper band
    bb_value = -1 → price is at the lower band
    """
    prices = data['Close']
    sma = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()

    # upper_band = sma + (rolling_std * num_std)
    # lower_band = sma - (rolling_std * num_std)

    bb_value = (prices - sma) / (num_std * rolling_std)
    return bb_value.to_frame("BB")

def atr(data, period=14, use_ema=False):

    data = data.copy()
    
    data['high_low'] = data['High'] - data['Low']
    data['high_close'] = abs(data['High'] - data['Close'].shift(1))
    data['low_close'] = abs(data['Low'] - data['Close'].shift(1))
    
    data['tr'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
    
    if use_ema:
        data['atr'] = data['tr'].ewm(span=period, adjust=False).mean()
    else:
        data['atr'] = data['tr'].rolling(window=period).mean()
    
    data['atr_ratio'] = data['atr'] / data['Close']
    return data['atr_ratio'].to_frame("ATR_Ratio")

def accumulation_distribution(data):

    data = data.copy()
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    range_hl = high - low
    range_hl = range_hl.replace(0, 1e-10)

    # A/D formula components
    mfm = ((close - low) - (high - close)) / range_hl  # Money Flow Multiplier
    mfv = mfm * volume                                 # Money Flow Volume
    ad = mfv.cumsum()                                  # Accumulation/Distribution Line

    ad_norm = (ad - ad.min()) / (ad.max() - ad.min())
    return ad_norm.to_frame("AD")
    
def rsi(data, window=14):
    """
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    """
    prices = data['Close']
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values.to_frame("RSI")

def macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    MACD = EMA(12) - EMA(26)
    Signal = EMA(9) of MACD
    Histogram = macd - signal
    """
    prices = data['Close']
    ema_fast = prices.ewm(span=fast_period).mean()
    ema_slow = prices.ewm(span=slow_period).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period).mean()
    macd_histogram = macd_line - signal_line

    return macd_histogram.to_frame("MACD")

def ema_ratio(data, window=10):
    """
    ema_ratio = price/EMA(10)
    """
    prices = data['Close']
    ema = prices.ewm(span=window).mean()
    return (prices / (ema + 1e-9)).to_frame("EMA_ratio")

def cci(data, window=20):
    """
    CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
    where Typical Price = (Adj High + Adj Low + Close) / 3
    """
    adj_close = data['Close']
    high = data['High']
    low = data['Low']
    close = data['Close']
    adj_factor = adj_close/close
    adj_high = high * adj_factor
    adj_low = low * adj_factor

    tp = (adj_high + adj_low + adj_close)/3
    sma = tp.rolling(window=window).mean()
    mean_deviation = tp.rolling(window=window).apply(
        lambda x: (x - x.mean()).abs().mean()
    )

    cci_values = (tp - sma) / (0.015 * mean_deviation)
    return cci_values.to_frame("CCI")

def obv(data, window=20):
    """
    Calculates OBV and then normalizes it using a Rolling Z-Score
    to make it stationary and scale-compatible with Log Returns.
    """
    # 1. Standard OBV Calculation
    close_diff = data['Close'].diff()
    direction = np.where(close_diff > 0, 1, -1)
    direction[close_diff == 0] = 0
    
    volume_flow = direction * data['Volume'] 
    obv_raw = volume_flow.cumsum()
    
    # 2. Make it Stationary (Rolling Z-Score)
    # (Value - Mean) / Std
    rolling_mean = obv_raw.rolling(window=window).mean()
    rolling_std = obv_raw.rolling(window=window).std()
    
    # Add epsilon to avoid division by zero
    obv_stationary = (obv_raw - rolling_mean) / (rolling_std + 1e-9)
    
    return obv_stationary.to_frame("OBV")

def get_all_data(start_date="2015-01-01", end_date="2024-01-01"):

    tickers = pd.read_csv('tickers.csv')['ticker'].tolist()
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        group_by='ticker',
        threads=True,
        auto_adjust=True,
    )

    all_data = []
    for ticker in tickers:

        ticker_df = raw[ticker].copy()
        ticker_df = ticker_df.reset_index()        
        ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
        ticker_df.set_index('Date', inplace=True)
        ticker_df = ticker_df[~ticker_df.index.duplicated(keep='first')]

        if isinstance(ticker_df['Close'], pd.DataFrame):
            ticker_df['Close'] = ticker_df['Close'].iloc[:, 0]
        if isinstance(ticker_df['Volume'], pd.DataFrame):
            ticker_df['Volume'] = ticker_df['Volume'].iloc[:, 0]
        if isinstance(ticker_df['High'], pd.DataFrame):
            ticker_df['High'] = ticker_df['High'].iloc[:, 0]
        if isinstance(ticker_df['Low'], pd.DataFrame):
            ticker_df['Low'] = ticker_df['Low'].iloc[:, 0]
        if isinstance(ticker_df['Open'], pd.DataFrame):
            ticker_df['Open'] = ticker_df['Open'].iloc[:, 0]

        ticker_df['Change'] = ticker_df['Close'].pct_change(fill_method=None)
        RSI = rsi(data=ticker_df)
        MACD = macd(data=ticker_df)
        EMA = ema_ratio(data=ticker_df)
        CCI = cci(data=ticker_df)
        OBV = obv(data=ticker_df)
        BB = bollinger_bands(data=ticker_df)
        ATR = atr(data=ticker_df)
        AD = accumulation_distribution(data=ticker_df)
        ticker_df.drop(columns=['Dividends', 'Stock Splits', 'Adj Close'], inplace=True, errors='ignore')
        ticker_df = pd.concat([ticker_df, RSI, MACD, EMA, CCI, OBV, BB, ATR, AD], axis=1)
        ticker_df['Target'] = ticker_df['Close'].pct_change(fill_method=None).shift(-1)
        cols_to_log = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols_to_log:
            if col in ticker_df.columns:
                current_val = ticker_df[col] + 1e-9
                prev_val = ticker_df[col].shift(1) + 1e-9
                ticker_df[col] = np.log(current_val / prev_val)

        ticker_df['Ticker'] = ticker
        ticker_df.replace([np.inf, -np.inf], 0, inplace=True)
        ticker_df.dropna(inplace=True)
        ticker_df['Target']= ticker_df['Target'].clip(lower=-0.5, upper=0.5)

        all_data.append(ticker_df)

    if not all_data:
        print("No ticker data fetched. Check tickers.csv and your internet connection.")
        return pd.DataFrame()

    combined = pd.concat(all_data)   # index will be Date (possibly duplicate dates across tickers)
    combined.to_csv('stock_data.csv', index=True)
    print("Saved combined data to stock_data.csv")
    print(combined.head())

    return combined

def get_ticker_data(ticker, is_log = True):

    data = yf.Ticker(ticker).history(period="100d")
    if data.empty:
        raise ValueError(f"No data found for {ticker}")
    
    data = data.reset_index()        
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    RSI = rsi(data=data)
    MACD = macd(data=data)
    EMA = ema_ratio(data=data)
    CCI = cci(data=data)
    OBV = obv(data=data)
    BB = bollinger_bands(data=data)
    ATR = atr(data=data)
    AD = accumulation_distribution(data=data)
    data.drop(columns=['Dividends', 'Stock Splits', 'Adj Close', 'Capital Gains'], inplace=True, errors='ignore')
    data['Change'] = data['Close'].pct_change(fill_method=None)
    cumulative = pd.concat([data, RSI, MACD, EMA, CCI, OBV, BB, ATR, AD], axis=1)

    if is_log:
        cols_to_log = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols_to_log:
            if col in cumulative.columns:
                current_val = cumulative[col] + 1e-9
                prev_val = cumulative[col].shift(1) + 1e-9
                cumulative[col] = np.log(current_val/prev_val)
    cumulative.dropna(inplace=True)
    return cumulative.astype(float)

if __name__ == "__main__":
    get_all_data()