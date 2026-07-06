# Time Series Forecasting for Portfolio Management Optimization

This project implements comprehensive time series forecasting models for portfolio optimization. It includes data extraction, exploratory data analysis, and multiple forecasting models (ARIMA, SARIMA, LSTM) applied to financial assets.

## Project Overview

**Objective**: Apply time series forecasting techniques to optimize portfolio management strategies.

**Assets Analyzed**:
- **TSLA**: Tesla, Inc. (High-growth stock)
- **BND**: Vanguard Total Bond Market ETF (Bond fund)
- **SPY**: SPDR S&P 500 ETF Trust (Market index)

**Data Period**: January 1, 2015 - June 30, 2026

## Project Structure

```
├── notebooks/
│   ├── 01_eda_analysis.ipynb      # Task 1: EDA and data analysis
│   └── 02_forecasting_models.ipynb # Task 2: ARIMA/SARIMA/LSTM models
├── scripts/
│   ├── fetch_data.py              # Data extraction script
│   └── run_arima.py               # ARIMA forecasting script
├── data/
│   ├── raw/                       # Raw data storage
│   └── processed/                 # Processed data and outputs
├── src/
│   ├── data/                      # Data fetching and preprocessing
│   ├── models/                    # Forecasting models
│   ├── optimization/              # Portfolio optimization
│   ├── visualization/             # Plotting tools
│   └── utils/                      # Helper functions
├── tests/                         # Unit tests
├── .github/workflows/             # CI configuration
├── requirements.txt               # Dependencies
├── config.json                    # Configuration
└── README.md                      # This file
```

## Setup Instructions

### 1. Prerequisites

- Python 3.9 or higher
- pip package manager

### 2. Clone Repository

```bash
git clone https://github.com/abigiyacodehub/Time-Series-Forecasting-for-Portfolio-Management-Optimization.git
cd Time-Series-Forecasting-for-Portfolio-Management-Optimization
```

### 3. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Fetch Data

```bash
cd scripts
python fetch_data.py
```

### 6. Run Analysis

**Option A: Using Jupyter Notebooks**
```bash
jupyter notebook notebooks/01_eda_analysis.ipynb
jupyter notebook notebooks/02_forecasting_models.ipynb
```

**Option B: Using Command Line**
```bash
python scripts/run_arima.py --ticker SPY --days 30
```

**Option C: Using Streamlit Dashboard**
```bash
streamlit run src/visualization/dashboard.py
```

## Data Sources

### Yahoo Finance (yfinance API)
- Historical daily closing prices
- Date range: January 1, 2015 to June 30, 2026
- Assets: TSLA, BND, SPY

## Key Features

### Task 1: Data Extraction, Cleaning, and EDA

1. **Data Extraction**: yfinance API for historical price data
2. **Data Cleaning**: Missing values handling, type checking, duplicates
3. **EDA Visualizations**:
   - Closing prices over time
   - Daily percentage changes
   - Rolling mean and volatility (30-day window)
4. **Outlier Detection**: Z-score method (>3 std deviations)
5. **Stationarity Testing**: Augmented Dickey-Fuller (ADF) test
6. **Risk Metrics**: Value at Risk (VaR), Sharpe Ratio

### Task 2: Initial Forecasting Models

1. **Train/Test Split**: Chronological (time-based, not random)
2. **ARIMA Model**: Auto-parameter selection with AIC
3. **SARIMA Model**: Seasonal component testing
4. **LSTM Model**: Deep learning with windowed sequences
5. **Forecast Evaluation**: RMSE, MAE, MAPE metrics

## Model Documentation

### ARIMA (p, d, q)
- **p (AR order)**: Determined from PACF analysis
- **d (Integration)**: Differencing order from ADF test
- **q (MA order)**: Determined from ACF analysis

### SARIMA (p, d, q)(P, D, Q, s)
- Seasonal component with period s=5 (weekly)
- Additional seasonal parameters for financial patterns

### LSTM Neural Network
- Input layer: Windowed sequences (default 60 days)
- LSTM layers: [64, 32, 16] units
- Dropout: 0.2 for regularization
- Output: Single value forecast

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |
| AIC | Akaike Information Criterion (model selection) |
| Sharpe Ratio | Risk-adjusted return |

## Running Tests

```bash
pytest tests/ -v
```

## CI/CD

GitHub Actions workflow is configured for:
- Automated testing on push/pull requests
- Python version matrix (3.9, 3.10, 3.11)
- Code linting with flake8
- Test coverage reporting

## Dependencies

Key packages:
- `yfinance`: Stock data extraction
- `pandas`, `numpy`: Data manipulation
- `statsmodels`: ARIMA/SARIMA models, ADF test
- `scikit-learn`: Metrics, preprocessing
- `tensorflow` (optional): LSTM model
- `matplotlib`, `seaborn`, `plotly`: Visualization
- `streamlit`: Interactive dashboard

## Usage Examples

### Python API

```python
from src.data.stock_data import StockDataFetcher
from src.models.arima_model import ARIMAForecaster

# Fetch data
fetcher = StockDataFetcher(['TSLA', 'BND', 'SPY'], '2015-01-01', '2026-06-30')
prices = fetcher.fetch_data()
returns = fetcher.calculate_returns()

# Run metrics
stats = fetcher.get_summary_stats()

# Fit ARIMA
model = ARIMAForecaster(order=(1, 1, 1))
model.fit(prices['SPY'])
forecast = model.predict(steps=30)
```

### Command Line

```bash
# Forecast for specific ticker
python scripts/run_arima.py --ticker TSLA --days 60

# Run main analysis
python main.py --tickers TSLA,BND,SPY --forecast-days 30
```

## License

MIT License

## Authors

Portfolio Optimization Team

## Acknowledgments

- Yahoo Finance for providing free market data
- statsmodels for time series analysis tools
- TensorFlow/Keras for deep learning capabilities

---

**Last Updated**: January 2024
