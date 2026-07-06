"""
Stock data acquisition module using yfinance API.
Fetches historical stock prices for portfolio analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta


class StockDataFetcher:
    """Fetches and manages stock price data from Yahoo Finance."""

    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """
        Initialize the data fetcher.

        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self._data: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.DataFrame] = None

    def fetch_data(self, retry_attempts: int = 3) -> pd.DataFrame:
        """
        Fetch adjusted close prices for all tickers.

        Args:
            retry_attempts: Number of retry attempts for failed fetches

        Returns:
            DataFrame with datetime index and ticker columns

        Raises:
            ValueError: If no data could be fetched for any ticker
        """
        print(f"Fetching data for {len(self.tickers)} tickers...")

        data_dict = {}
        failed_tickers = []

        for ticker in self.tickers:
            for attempt in range(retry_attempts):
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=self.start_date, end=self.end_date)
                    if not hist.empty:
                        data_dict[ticker] = hist['Close']
                        print(f"  Fetched {ticker}: {len(hist)} records")
                        break
                    else:
                        if attempt == retry_attempts - 1:
                            print(f"  Warning: No data for {ticker} after {retry_attempts} attempts")
                            failed_tickers.append(ticker)
                except Exception as e:
                    if attempt == retry_attempts - 1:
                        print(f"  Error fetching {ticker} after {retry_attempts} attempts: {e}")
                        failed_tickers.append(ticker)
                    else:
                        print(f"  Retry {attempt + 1}/{retry_attempts} for {ticker}")

        # Check if any data was fetched
        if not data_dict:
            raise ValueError(f"Failed to fetch data for any tickers: {self.tickers}")

        if failed_tickers:
            print(f"\nWarning: Failed to fetch data for {len(failed_tickers)} ticker(s): {failed_tickers}")

        self._data = pd.DataFrame(data_dict)
        self._data.index = pd.to_datetime(self._data.index)

        # Handle missing values
        if self._data.isnull().any().any():
            print(f"Handling missing values...")
            self._data = self._data.fillna(method='ffill').fillna(method='bfill')

        print(f"\nSuccessfully fetched data for {len(self._data.columns)} tickers")
        print(f"Total observations: {len(self._data)}")

        return self._data

    def calculate_returns(self, method: str = 'log') -> pd.DataFrame:
        """
        Calculate daily returns from price data.

        Args:
            method: 'log' for log returns, 'simple' for simple returns

        Returns:
            DataFrame of daily returns
        """
        if self._data is None:
            raise ValueError("No data available. Call fetch_data() first.")

        if method == 'log':
            self._returns = np.log(self._data / self._data.shift(1)).dropna()
        else:
            self._returns = self._data.pct_change().dropna()

        return self._returns

    def get_summary_stats(self) -> pd.DataFrame:
        """
        Calculate summary statistics for each stock.

        Returns:
            DataFrame with mean return, volatility, and other metrics
        """
        if self._returns is None:
            raise ValueError("No returns data. Call calculate_returns() first.")

        stats = pd.DataFrame({
            'Mean Daily Return': self._returns.mean(),
            'Daily Volatility': self._returns.std(),
            'Annualized Return': self._returns.mean() * 252,
            'Annualized Volatility': self._returns.std() * np.sqrt(252),
            'Sharpe Ratio': (self._returns.mean() * 252) / (self._returns.std() * np.sqrt(252)),
            'Max Return': self._returns.max(),
            'Min Return': self._returns.min(),
            'Skewness': self._returns.skew(),
            'Kurtosis': self._returns.kurtosis()
        })

        return stats

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix of returns."""
        if self._returns is None:
            raise ValueError("No returns data. Call calculate_returns() first.")

        return self._returns.corr()

    def calculate_covariance_matrix(self, annualize: bool = True) -> pd.DataFrame:
        """
        Calculate covariance matrix of returns.

        Args:
            annualize: Whether to annualize the covariance

        Returns:
            Covariance matrix
        """
        if self._returns is None:
            raise ValueError("No returns data. Call calculate_returns() first.")

        cov_matrix = self._returns.cov()
        if annualize:
            cov_matrix *= 252

        return cov_matrix

    def add_technical_indicators(self, ticker: str) -> pd.DataFrame:
        """
        Add technical indicators for a specific ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with technical indicators
        """
        if self._data is None or ticker not in self._data.columns:
            raise ValueError(f"No price data for {ticker}")

        prices = self._data[ticker].copy()
        df = pd.DataFrame(index=prices.index)
        df['Price'] = prices

        # Moving averages
        df['SMA_20'] = prices.rolling(window=20).mean()
        df['SMA_50'] = prices.rolling(window=50).mean()
        df['EMA_12'] = prices.ewm(span=12, adjust=False).mean()
        df['EMA_26'] = prices.ewm(span=26, adjust=False).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = prices.rolling(window=20).mean()
        df['BB_Std'] = prices.rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

        # Volatility
        df['Volatility_20'] = prices.pct_change().rolling(window=20).std() * np.sqrt(252)

        return df.dropna()

    def save_data(self, filepath: str) -> None:
        """Save price data to CSV."""
        if self._data is not None:
            self._data.to_csv(filepath)
            print(f"Data saved to {filepath}")

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load price data from CSV."""
        self._data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.tickers = list(self._data.columns)
        return self._data

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get the price data."""
        return self._data

    @property
    def returns(self) -> Optional[pd.DataFrame]:
        """Get the returns data."""
        return self._returns
