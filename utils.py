import pandas as pd
import numpy as np

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

if __name__ == "__main__":
    analyze_data()