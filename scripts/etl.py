import os
from pathlib import Path

import yfinance as yf
import numpy as np
from scipy import stats
import pandas as pd
from pycoingecko import CoinGeckoAPI

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("FINWISE_DATA_DIR", REPO_ROOT / "data")).expanduser().resolve()
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Extract historical stock data from Yahoo Finance and cryptocurrency data from CoinGecko
def fetch_and_save_stocks(tickers, out_dir=RAW_DIR):
    for ticker in tickers:
        data = yf.Ticker(ticker).history(period="max")
        data.to_csv(out_dir / f"{ticker}_daily.csv")

def fetch_and_save_coins(coins, days=365, out_file=RAW_DIR / "crypto_prices.csv"):
    cg = CoinGeckoAPI()
    all_prices = []
    for coin_id in coins:
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency="usd", days=days)
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
        prices["coin"] = coin_id
        all_prices.append(prices)
    pd.concat(all_prices).to_csv(out_file, index=False)



tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"]
coins = [
    "bitcoin",
    "ethereum",
    "tether",
    "ripple",
    "binancecoin",
    "solana",
    "dogecoin",
    "tron",
    "usd-coin",
]

fetch_and_save_stocks(tickers)
fetch_and_save_coins(coins)

# Load data
AAPL = pd.read_csv(RAW_DIR / "AAPL_daily.csv")
GOOGL = pd.read_csv(RAW_DIR / "GOOGL_daily.csv")
MSFT = pd.read_csv(RAW_DIR / "MSFT_daily.csv")
AMZN = pd.read_csv(RAW_DIR / "AMZN_daily.csv")
TSLA = pd.read_csv(RAW_DIR / "TSLA_daily.csv")
META = pd.read_csv(RAW_DIR / "META_daily.csv")
crypto = pd.read_csv(RAW_DIR / "crypto_prices.csv")

fin_check = [AAPL, GOOGL, MSFT, AMZN, TSLA, META, crypto]

for i in fin_check:
    z = np.abs(stats.zscore(i.select_dtypes(include=[np.number])))
    outliers = (z > 3).any(axis=1)
    outliers_rate = outliers.mean()
    print(f"Outliers rate: {outliers_rate}")
    print(f"Info: {i.info()}")
    print(f"Duplicated rows: {i.duplicated().sum()}")
    print(f"Missing values row: {i.isnull().any(axis=1).sum()}")

"""
    AAPL: 
        + Info: Ticker Date, Close, High, Low, Open, Volume
        + No duplicated rows, missing values
        + Checking outliers with Z-score -> No outliers detected
    GOOGL: 
        + Info: Ticker Date, Close, High, Low, Open, Volume
        + No duplicated rows, missing values
        + Checking outliers with Z-score -> No outliers detected
    MSFT: 
        + Info: Ticker Date, Close, High, Low, Open, Volume
        + No duplicated rows, missing values
        + Checking outliers with Z-score -> No outliers detected
    AMZN:
        + Info: Ticker Date, Close, High, Low, Open, Volume
        + No duplicated rows, missing values
        + Checking outliers with Z-score -> No outliers detected
    TSLA:
        + Info: Ticker Date, Close, High, Low, Open, Volume
        + No duplicated rows, missing values
        + Checking outliers with Z-score -> No outliers detected
    META:
        + Info: Ticker Date, Close, High, Low, Open, Volume
        + No duplicated rows, missing values
        + Checking outliers with Z-score -> No outliers detected
    crypto: 
        + Info: timestamp, price, coin
        + No duplicated rows, missing values
        + Checking outliers with Z-score -> No outliers detected
"""

AAPL = AAPL.dropna().reset_index(drop=True)
GOOGL = GOOGL.dropna().reset_index(drop=True)
MSFT = MSFT.dropna().reset_index(drop=True)
AMZN = AMZN.dropna().reset_index(drop=True)
TSLA = TSLA.dropna().reset_index(drop=True)
META = META.dropna().reset_index(drop=True)

# Feature Engineering with DateTime and Numerical features
crypto["timestamp"] = pd.to_datetime(crypto["timestamp"], errors="coerce", utc=True)
AAPL["Date"] = pd.to_datetime(AAPL["Date"], errors="coerce", utc=True)
GOOGL["Date"] = pd.to_datetime(GOOGL["Date"], errors="coerce", utc=True)
MSFT["Date"] = pd.to_datetime(MSFT["Date"], errors="coerce", utc=True)
AMZN["Date"] = pd.to_datetime(AMZN["Date"], errors="coerce", utc=True)
TSLA["Date"] = pd.to_datetime(TSLA["Date"], errors="coerce", utc=True)
META["Date"] = pd.to_datetime(META["Date"], errors="coerce", utc=True)

# Save processed data
AAPL.to_csv(PROCESSED_DIR / "AAPL_daily_processed.csv", index=False)
GOOGL.to_csv(PROCESSED_DIR / "GOOGL_daily_processed.csv", index=False)
MSFT.to_csv(PROCESSED_DIR / "MSFT_daily_processed.csv", index=False)
AMZN.to_csv(PROCESSED_DIR / "AMZN_daily_processed.csv", index=False)
TSLA.to_csv(PROCESSED_DIR / "TSLA_daily_processed.csv", index=False)
META.to_csv(PROCESSED_DIR / "META_daily_processed.csv", index=False)
crypto.to_csv(PROCESSED_DIR / "crypto_prices_processed.csv", index=False)
