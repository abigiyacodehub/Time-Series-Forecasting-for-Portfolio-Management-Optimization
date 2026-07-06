"""
Data preprocessing module for time series analysis.
Handles cleaning, normalization, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List


class DataPreprocessor:
    """Preprocesses time series data for forecasting models."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize preprocessor.

        Args:
            data: DataFrame with datetime index
        """
        self.data = data.copy()
        self.scaler: Optional[StandardScaler] = None
        self.minmax_scaler: Optional[MinMaxScaler] = None

    def handle_missing_values(self, method: str = 'ffill') -> pd.DataFrame:
        """
        Handle missing values in the data.

        Args:
            method: 'ffill', 'bfill', 'interpolate', or 'drop'

        Returns:
            DataFrame with missing values handled
        """
        if method == 'ffill':
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        elif method == 'bfill':
            self.data = self.data.fillna(method='bfill').fillna(method='ffill')
        elif method == 'interpolate':
            self.data = self.data.interpolate(method='time')
        elif method == 'drop':
            self.data = self.data.dropna()

        return self.data

    def remove_outliers(self, method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from the data.

        Args:
            method: 'zscore' or 'iqr'
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        if method == 'zscore':
            z_scores = np.abs((self.data - self.data.mean()) / self.data.std())
            self.data = self.data.mask(z_scores > threshold, self.data.median())
        elif method == 'iqr':
            Q1 = self.data.quantile(0.25)
            Q3 = self.data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            self.data = self.data.clip(lower_bound, upper_bound, axis=1)

        return self.data

    def normalize(self, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize the data.

        Args:
            method: 'standard' or 'minmax'

        Returns:
            Normalized DataFrame
        """
        if method == 'standard':
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(self.data)
        else:
            self.minmax_scaler = MinMaxScaler()
            scaled_data = self.minmax_scaler.fit_transform(self.data)

        self.data = pd.DataFrame(scaled_data, index=self.data.index,
                                  columns=self.data.columns)
        return self.data

    def create_sequences(self, sequence_length: int, target_column: str = None,
                         target_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series models (e.g., LSTM).

        Args:
            sequence_length: Number of time steps in each sequence
            target_column: Column to predict (if None, uses all columns)
            target_horizon: How many steps ahead to predict

        Returns:
            Tuple of (X, y) arrays
        """
        data_values = self.data.values
        X, y = [], []

        for i in range(len(data_values) - sequence_length - target_horizon + 1):
            X.append(data_values[i:(i + sequence_length)])

            if target_column:
                idx = self.data.columns.get_loc(target_column)
                y.append(data_values[i + sequence_length + target_horizon - 1, idx])
            else:
                y.append(data_values[i + sequence_length + target_horizon - 1])

        return np.array(X), np.array(y)

    def split_train_test(self, test_size: float = 0.2,
                         shuffle: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.

        Args:
            test_size: Proportion of data for testing
            shuffle: Whether to shuffle (not recommended for time series)

        Returns:
            Tuple of (train, test) DataFrames
        """
        if shuffle:
            return train_test_split(self.data, test_size=test_size, shuffle=shuffle)
        else:
            split_idx = int(len(self.data) * (1 - test_size))
            return self.data.iloc[:split_idx], self.data.iloc[split_idx:]

    def add_lagged_features(self, lags: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Add lagged features to the data.

        Args:
            lags: List of lag periods

        Returns:
            DataFrame with lagged features
        """
        for col in self.data.columns:
            for lag in lags:
                self.data[f'{col}_lag_{lag}'] = self.data[col].shift(lag)

        return self.data.dropna()

    def add_rolling_features(self, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Add rolling window statistics.

        Args:
            windows: List of window sizes

        Returns:
            DataFrame with rolling features
        """
        for col in self.data.columns:
            for window in windows:
                self.data[f'{col}_rolling_mean_{window}'] = self.data[col].rolling(window).mean()
                self.data[f'{col}_rolling_std_{window}'] = self.data[col].rolling(window).std()

        return self.data.dropna()

    def add_time_features(self) -> pd.DataFrame:
        """
        Add time-based features derived from the datetime index.

        Returns:
            DataFrame with time features
        """
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['day_of_month'] = self.data.index.day
        self.data['month'] = self.data.index.month
        self.data['quarter'] = self.data.index.quarter
        self.data['year'] = self.data.index.year
        self.data['week_of_year'] = self.data.index.isocalendar().week

        return self.data

    def create_walk_forward_splits(self, n_splits: int = 5,
                                   test_size: int = 30) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward validation splits.

        Args:
            n_splits: Number of splits
            test_size: Size of test set for each split

        Returns:
            List of (train, test) DataFrame tuples
        """
        splits = []
        total_size = len(self.data)
        split_size = (total_size - test_size) // n_splits

        for i in range(n_splits):
            train_end = split_size * (i + 1) + test_size * i
            test_end = train_end + test_size

            if test_end <= total_size:
                train = self.data.iloc[:train_end]
                test = self.data.iloc[train_end:test_end]
                splits.append((train, test))

        return splits

    def inverse_transform(self, data: np.ndarray,
                          columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Inverse transform normalized data.

        Args:
            data: Normalized data array
            columns: Specific columns to inverse transform

        Returns:
            Original scale data
        """
        if self.scaler:
            return self.scaler.inverse_transform(data)
        elif self.minmax_scaler:
            return self.minmax_scaler.inverse_transform(data)
        return data
