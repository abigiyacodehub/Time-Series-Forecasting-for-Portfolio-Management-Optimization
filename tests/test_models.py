"""Tests for forecasting models."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_series():
    """Generate sample time series for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    # Random walk with drift
    values = 100 + np.cumsum(np.random.randn(200) * 2 + 0.1)
    return pd.Series(values, index=dates)


class TestARIMAForecaster:
    """Tests for ARIMA model."""

    def test_stationarity_check(self, sample_series):
        """Test stationarity check."""
        from src.models.arima_model import ARIMAForecaster

        arima = ARIMAForecaster()
        result = arima.check_stationarity(sample_series)

        assert 'Is Stationary' in result
        assert 'p-value' in result

    def test_fit_and_predict(self, sample_series):
        """Test ARIMA fitting and prediction."""
        from src.models.arima_model import ARIMAForecaster

        arima = ARIMAForecaster(order=(1, 1, 1))
        arima.fit(sample_series)

        forecast = arima.predict(steps=10)

        assert len(forecast) == 10
        assert all(not np.isnan(v) for v in forecast)


class TestDataPreprocessor:
    """Tests for data preprocessing."""

    def test_handle_missing_values(self, sample_series):
        """Test missing value handling."""
        from src.data.preprocessor import DataPreprocessor

        df = pd.DataFrame({'price': sample_series})
        df.loc[df.index[5], 'price'] = np.nan

        preprocessor = DataPreprocessor(df)
        result = preprocessor.handle_missing_values()

        assert not result.isnull().any().any()

    def test_normalize(self, sample_series):
        """Test normalization."""
        from src.data.preprocessor import DataPreprocessor

        df = pd.DataFrame({'price': sample_series})
        preprocessor = DataPreprocessor(df)

        normalized = preprocessor.normalize(method='standard')

        # Check approximate mean=0, std=1
        assert abs(normalized['price'].mean()) < 0.1
        assert abs(normalized['price'].std() - 1) < 0.1

    def test_train_test_split(self, sample_series):
        """Test data splitting."""
        from src.data.preprocessor import DataPreprocessor

        df = pd.DataFrame({'price': sample_series})
        preprocessor = DataPreprocessor(df)

        train, test = preprocessor.split_train_test(test_size=0.2)

        assert len(train) + len(test) == len(df)
        assert len(test) == int(len(df) * 0.2)


class TestStockDataFetcher:
    """Tests for stock data fetcher."""

    def test_calculate_returns(self):
        """Test return calculation."""
        from src.data.stock_data import StockDataFetcher

        # Create mock data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.randn(100)),
            'MSFT': 200 + np.cumsum(np.random.randn(100) * 2)
        }, index=dates)

        fetcher = StockDataFetcher.__new__(StockDataFetcher)
        fetcher._data = data

        returns = fetcher.calculate_returns()

        assert len(returns) == len(data) - 1
        assert all(abs(returns.values).max() < 0.5)

    def test_summary_stats(self):
        """Test summary statistics calculation."""
        from src.data.stock_data import StockDataFetcher

        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.randn(100)),
        }, index=dates)

        fetcher = StockDataFetcher.__new__(StockDataFetcher)
        fetcher._data = data
        fetcher._returns = data.pct_change().dropna()

        stats = fetcher.get_summary_stats()

        assert 'Mean Daily Return' in stats.columns
        assert 'Sharpe Ratio' in stats.columns
