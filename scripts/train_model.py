import os
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

plt.style.use("ggplot")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("FINWISE_DATA_DIR", REPO_ROOT / "data")).expanduser().resolve()
PROCESSED_DIR = DATA_ROOT / "processed"

AAPL = pd.read_csv(PROCESSED_DIR / "AAPL_daily_processed.csv")
GOOGL = pd.read_csv(PROCESSED_DIR / "GOOGL_daily_processed.csv")
MSFT = pd.read_csv(PROCESSED_DIR / "MSFT_daily_processed.csv")
AMZN = pd.read_csv(PROCESSED_DIR / "AMZN_daily_processed.csv")
TSLA = pd.read_csv(PROCESSED_DIR / "TSLA_daily_processed.csv")
META = pd.read_csv(PROCESSED_DIR / "META_daily_processed.csv")

#crypto = pd.read_csv("..data/processed/crypto_prices_processed.csv")
#bitcoin_data = crypto[crypto["coin"] == "bitcoin"]
#ethereum_data = crypto[crypto["coin"] == "ethereum"]
#tether_data = crypto[crypto["coin"] == "tether"]
#xrp_data = crypto[crypto["coin"] == "ripple"]
#usdcoin_data = crypto[crypto["coin"] == "usd-coin"]

### Split features and target
fig, ax = plt.subplots(3, 2, figsize=(12, 10))
ax[0, 0].plot(AAPL["Date"], AAPL["Close"])
ax[0, 0].set_title("AAPL Close Price Over Time")
ax[0, 1].plot(GOOGL["Date"], GOOGL["Close"])
ax[0, 1].set_title("GOOGL Close Price Over Time")
ax[1, 0].plot(MSFT["Date"], MSFT["Close"])
ax[1, 0].set_title("MSFT Close Price Over Time")
ax[1, 1].plot(AMZN["Date"], AMZN["Close"])
ax[1, 1].set_title("AMZN Close Price Over Time")
ax[2, 0].plot(TSLA["Date"], TSLA["Close"])
ax[2, 0].set_title("TSLA Close Price Over Time")
ax[2, 1].plot(META["Date"], META["Close"])
ax[2, 1].set_title("META Close Price Over Time")
plt.tight_layout()
plt.savefig("../reports/figures/stock_prices_over_time.png")


### Preprocess and training pipeline
yahoo_numerical_features = ["Close", "High", "Low", "Open", "Volume"]
yahoo_time = ["Ticker Date"]
crypto_numerical_features = ["price"]
crypto_categorical_features = ["coin"]
crypto_time = ["timestamp"]

yahoo_numerical_transformer = Pipeline(
    steps=[("impute", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler())]
)

crypto_numerical_transformer = Pipeline(
    steps=[("impute", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler())]
)

crypto_categorical_transformer = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

yahoo_transformer = ColumnTransformer(
    transformers=[("num", yahoo_numerical_transformer, yahoo_numerical_features)]
)

crypto_transformer = ColumnTransformer(
    transformers=[
        ("num", crypto_numerical_transformer, crypto_numerical_features),
        ("cat", crypto_categorical_transformer, crypto_categorical_features),
    ]
)

#### Using top models for time series forecasting and fine tuning hyperparameters in the next steps
prophet_model = Prophet()
arima_model = ARIMA(order=(5, 1, 0))
sarimax_model = SARIMAX(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
rf_mode = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(
    objective="reg:squarederror", n_estimators=100, random_state=42
)

### Fine-tuning models' hyperparameters
"""
    Prophet: changepoint_prior_scale, changepoint_range, holidays_prior_scale, seasonality_mode, growth
    XGBoost: min_child_weight, gamma, subsample, colsample_bytree, max_depth
    Random Forest: n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf
    Arima and Sarimax: use default parameters for now
"""
prophet_paramsGrid = {
    "changepoint_prior_scale": [
        [0.005, 0.01, 0.05, 0.5, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ],
    "changepoint_range": [0.8, 0.9],
    "holidays_prior_scale": [
        [0.005, 0.01, 0.05, 0.5, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ],
    "seasonality_mode": ["multiplicative", "additive"],
    "growth": ["linear", "logistic"],
}

xgb_paramsGrid = {
    "min_child_weight": [1, 5, 10],
    "gamma": [0.5, 1, 1.5, 2, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "max_depth": [3, 4, 5],
}

rf_paramsGrid = {
    "n_estimators": [100, 200, 300],
    "max_features": ["auto", "sqrt"],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
