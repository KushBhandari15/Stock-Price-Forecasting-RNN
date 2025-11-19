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

def obv(data):
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])

    return pd.Series(obv, index=data.index, name="OBV")

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
            ticker_df['Close'] = ticker_df['Close'].iloc[:,0]
        ticker_df['Change'] = ticker_df['Close'].pct_change(fill_method=None)
        ticker_df['Ticker'] = ticker
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

if __name__ == "__main__":
    get_all_data()