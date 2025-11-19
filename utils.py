import pandas as pd
import numpy as np
import requests
from io import StringIO

def analyze_data():

    data = pd.read_csv('stock_data.csv')
    target = data['Target'].dropna()
    target_mean = target.mean()
    target_std = target.std()
    target_skew = target.skew()
    target_kurtosis = target.kurtosis()

    print(f"Total Samples: {len(target)}")
    print(f"Target Mean: {target_mean}")
    print(f"Target Std Dev: {target_std}")
    print(f"Target Skewness: {target_skew}")
    print(f"Target Kurtosis: {target_kurtosis}")

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)

    if "S&P 500" not in response.text:
        print("Error: The page content does not look right. You might be blocked.")
        print("First 200 chars of content:", response.text[:200])
        return

    try:
        tables = pd.read_html(StringIO(response.text), attrs={'id': 'constituents'})
        df = tables[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        df = df[['Symbol', 'GICS Sector', 'GICS Sub-Industry']]
        df.rename(columns={'Symbol': 'ticker', 'GICS Sector': 'sector', 'GICS Sub-Industry': 'industry'}, inplace=True)
        df.to_csv('tickers.csv', index=False)
        print(df.head())

    except ValueError as e:
        print(f"Could not find the table. Error: {e}")

if __name__ == "__main__":
    get_sp500_tickers()