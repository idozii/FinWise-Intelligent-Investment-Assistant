import yfinance as yf
import pandas as pd
from pycoingecko import CoinGeckoAPI

### Extract historical stock data from Yahoo Finance and cryptocurrency data from CoinGecko
tickers = ["AAPL", "MSFT", "GOOGL"]
for ticker in tickers:
    data = yf.download(ticker, start="2020-01-01", end="2025-12-31", interval="1d")
    data.to_csv(f"../data/raw/{ticker}_daily.csv")


cg = CoinGeckoAPI()
coins = ["bitcoin", "ethereum", "litecoin"]
all_prices = []
for coin_id in coins:
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency="usd", days=365)
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
    prices["coin"] = coin_id
    all_prices.append(prices)
pd.concat(all_prices).to_csv("../data/raw/crypto_prices.csv", index=False)

### Process the data
AAPL = pd.read_csv("../data/raw/AAPL_daily.csv")
GOOGL = pd.read_csv("../data/raw/GOOGL_daily.csv")
MSFT = pd.read_csv("../data/raw/MSFT_daily.csv")
crypto = pd.read_csv("../data/raw/crypto_prices.csv")

print(GOOGL.head())
print(GOOGL.info())
print(GOOGL.isna().mean())

"""
    AAPL: 1 row has NaN values -> drop na

"""
