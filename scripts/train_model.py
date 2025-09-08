import pandas as pd
import numpy as np
from scipy import stats
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

# Clean and load data
AAPL = pd.read_csv("../data/raw/AAPL_daily.csv")
GOOGL = pd.read_csv("../data/raw/GOOGL_daily.csv")
MSFT = pd.read_csv("../data/raw/MSFT_daily.csv")
crypto = pd.read_csv("../data/raw/crypto_prices.csv")

fin_check = [AAPL, GOOGL, MSFT, crypto]

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
        + Info: Price, Close, High, Low, Open, Volume
        + 1 row has missing values -> drop na
        + No duplicated rows
        + Checking outliers with Z-score -> No outliers detected
    GOOGL: 
        + Info: Price, Close, High, Low, Open, Volume
        + 1 row has missing values -> drop na
        + No duplicated rows
        + Checking outliers with Z-score -> No outliers detected
    MSFT: 
        + Info: Price, Close, High, Low, Open, Volume
        + 1 row has missing values -> drop na
        + No duplicated rows
        + Checking outliers with Z-score -> No outliers detected
    crypto: 
        + Info: timestamp, price, coin
        + No missing values
        + No duplicated rows
        + Checking outliers with Z-score -> No outliers detected
"""

AAPL = AAPL.dropna()
AAPL = AAPL.reset_index(drop=True)
GOOGL = GOOGL.dropna()
GOOGL = GOOGL.reset_index(drop=True)
MSFT = MSFT.dropna()
MSFT = MSFT.reset_index(drop=True)

### Feature Engineering with DateTime and Numerical features
crypto["timestamp"] = pd.to_datetime(crypto["timestamp"], errors="coerce")
AAPL["Ticker Date"] = pd.to_datetime(AAPL["Ticker Date"], errors="coerce")
GOOGL["Ticker Date"] = pd.to_datetime(GOOGL["Ticker Date"], errors="coerce")
MSFT["Ticker Date"] = pd.to_datetime(MSFT["Ticker Date"], errors="coerce")

### Save processed data
AAPL.to_csv("../data/processed/AAPL_daily_processed.csv", index=False)
GOOGL.to_csv("../data/processed/GOOGL_daily_processed.csv", index=False)
MSFT.to_csv("../data/processed/MSFT_daily_processed.csv", index=False)
crypto.to_csv("../data/processed/crypto_prices_processed.csv", index=False)

### Split features and target


### Preprocess and training pipeline
yahoo_numerical_features = ["Close", "High", "Low", "Open", "Volume"]
yahoo_time = ["Ticker Date"]
crypto_numerical_features = ["price"]
crypto_categorical_features = ["coin"]
crypto_time = ["timestamp"]

yahoo_numerical_transformer = Pipeline(
    steps=[("impute", KNNImputer(neighbors=5)), ("scaler", StandardScaler())]
)

crypto_numerical_transformer = Pipeline(
    steps=[("impute", KNNImputer(neighbors=5)), ("scaler", StandardScaler())]
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

### Save the model
