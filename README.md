# FinWise-Intelligent-Investment-Assistant

## Goals

Predict stock/ crypto prices + give recommendations with dashboard visualization.

## Implemented Tech Stack (as of October 2026)

- **Backend:** Python + FastAPI
- **Frontend:** static HTML/CSS/JavaScript, Chart.js, Plotly
- **Forecasts exposed by API:** ARIMA(5,1,0), Holt-Winters (damped trend), linear trend, moving-average
- **Data:** processed local CSVs; Yahoo Finance and CoinGecko retrieval when supported
- **Deployment:** Docker / Uvicorn / static FastAPI mount

**Not integrated into the live API:** LSTM, Prophet, Django, React/React Native and PostgreSQL. Some appear in an exploratory training script or earlier project plans, but should not be described as shipped features.

## Research Evaluation — New

Offline, leakage-safe expanding-window walk-forward comparison against a naive last-observation baseline, with per-horizon MAE, RMSE, directional-accuracy diagnostics, explicit statistical-model fallbacks, dataset SHA-256 provenance and automated tests.

Run `python -m pytest -q tests/test_backtest.py`. See [research/README.md](research/README.md) for the documented protocol and stock/crypto CLI examples.

**Research status:** synthetic tests validate evaluator logic; no out-of-sample market performance, empirical coverage of 95% intervals, trading return or investment recommendation has been established. FinWise is an educational decision-support prototype, not financial advice.

## Datasets

- Yahoo Finance: yfinance Python API
- Alpha Vantage: API key required
- CoinGecko: crypto market data

## Data Storage

Large CSV datasets should not be committed to GitHub.

- The repo now expects market data under a configurable base directory via `FINWISE_DATA_DIR`.
- Default local path is `./data`, but you can point it anywhere on disk, for example `/srv/finwise-data`.
- Required layout:

```text
FINWISE_DATA_DIR/
	raw/
	processed/
```

- The repo ignores `data/raw/*.csv` and `data/processed/*.csv` by default.
- If CSVs were already added to git index, untrack them once with:

```bash
git rm --cached data/raw/*.csv data/processed/*.csv
```

Then keep the files locally or move them to external storage such as a mounted disk, S3-backed sync folder, or a server path outside the repo.

## Run Web Dashboard

### 1) Start backend API

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export FINWISE_DATA_DIR=/absolute/path/to/your/data
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 2) Open frontend

Serve the frontend folder with a simple static server:

```bash
cd frontend
python -m http.server 5500
```

If you need to regenerate data first:

```bash
cd scripts
export FINWISE_DATA_DIR=/absolute/path/to/your/data
python etl.py
```

Open this URL in your browser:

- [http://127.0.0.1:5500](http://127.0.0.1:5500)

The dashboard will load stock symbols from your processed data and show:

- Historical close-price charts
- Line/Candlestick chart mode for stocks
- Multiple forecast model choices: ARIMA, Holt-Winters, Linear Trend, Moving Average
- 7-step point forecasts with estimated upper/lower intervals in the prediction table (interval coverage not yet calibrated)
- Prediction table and quick metrics
- Crypto dashboard tab using processed coin data plus a larger built-in asset universe

### API summary

- /api/forecast-models
- /api/stocks
- /api/stocks/{symbol}/history
- /api/stocks/{symbol}/forecast
- /api/crypto/coins
- /api/crypto/{coin}/history
- /api/crypto/{coin}/forecast

## Production Deployment

In production, the backend serves the frontend static files from the same origin.
That means users do not need to manually set API URLs.

### Option 1: Run directly on a server

```bash
cd /path/to/FinWise-Intelligent-Investment-Assistant
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
export FINWISE_DATA_DIR=/srv/finwise-data
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 2
```

Open:

- [http://YOUR_SERVER_IP:8000](http://YOUR_SERVER_IP:8000)

### Option 2: Docker

Build image:

```bash
docker build -t finwise:latest .
```

Run container:

```bash
docker run -d \
	--name finwise-app \
	-p 8000:8000 \
	-v /srv/finwise-data:/app/data \
	finwise:latest
```

Open:

- [http://YOUR_SERVER_IP:8000](http://YOUR_SERVER_IP:8000)

## Contributors

- 🖥️ [idozii](https://github.com/idozii)
