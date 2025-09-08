import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
GOOGL = GOOGL.dropna()
MSFT = MSFT.dropna()

### Feature Engineering with DateTime and Numerical features
crypto["timestamp"] = pd.to_datetime(crypto["timestamp"], unit="s")

### Split features and target


### Preprocess and training pipeline
yahoo_numerical_features = ["Close", "High", "Low", "Open", "Volume"]
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

### Save the model
