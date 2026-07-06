"""
Facebook Prophet forecasting model for time series prediction.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from typing import Dict, Any, Optional, List
import joblib
from pathlib import Path
import json


class ProphetForecaster:
    """Prophet-based time series forecaster."""

    def __init__(self, growth: str = 'linear',
                 seasonality_mode: str = 'multiplicative',
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 changepoint_prior_scale: float = 0.05):
        """
        Initialize Prophet forecaster.

        Args:
            growth: 'linear' or 'logistic'
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality
            changepoint_prior_scale: Flexibility of trend changes
        """
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale

        self.model: Optional[Prophet] = None
        self._cap: Optional[float] = None
        self._floor: Optional[float] = None

    def prepare_data(self, series: pd.Series) -> pd.DataFrame:
        """
        Prepare data for Prophet.

        Args:
            series: Time series with datetime index

        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        return df

    def set_growth_bounds(self, cap: Optional[float] = None,
                          floor: Optional[float] = None) -> None:
        """
        Set growth bounds for logistic growth.

        Args:
            cap: Maximum value (capacity)
            floor: Minimum value
        """
        self._cap = cap
        self._floor = floor

    def fit(self, data: pd.Series,
            regressors: Optional[pd.DataFrame] = None) -> 'ProphetForecaster':
        """
        Fit Prophet model to data.

        Args:
            data: Time series to model
            regressors: Additional regressors DataFrame

        Returns:
            Fitted forecaster
        """
        df = self.prepare_data(data)

        if self.growth == 'logistic':
            if self._cap is None:
                self._cap = df['y'].max() * 1.2
            if self._floor is None:
                self._floor = df['y'].min() * 0.8
            df['cap'] = self._cap
            df['floor'] = self._floor

        self.model = Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale
        )

        if regressors is not None:
            for col in regressors.columns:
                self.model.add_regressor(col)
                df[col] = regressors[col].values

        with df['ds'] as pd.NamedAgg(column='ds', aggfunc='first'):
            pass

        self.model.fit(df)
        print(f"Prophet model fitted successfully")

        return self

    def predict(self, periods: int = 30,
                freq: str = 'D',
                include_history: bool = False,
                regressors_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate forecasts.

        Args:
            periods: Number of periods to forecast
            freq: Frequency string ('D', 'W', 'M')
            include_history: Include historical data
            regressors_df: Future regressor values

        Returns:
            DataFrame with forecasts
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=include_history
        )

        if self.growth == 'logistic':
            future['cap'] = self._cap
            future['floor'] = self._floor

        if regressors_df is not None:
            for col in regressors_df.columns:
                future[col] = regressors_df[col].values

        forecast = self.model.predict(future)

        return forecast

    def cross_validate(self, initial: str = '365 days',
                       period: str = '90 days',
                       horizon: str = '30 days') -> pd.DataFrame:
        """
        Perform cross-validation.

        Args:
            initial: Initial training period
            period: Period between cutoffs
            horizon: Forecast horizon

        Returns:
            DataFrame with cross-validation results
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        df = self.model.history

        cv_results = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )

        return cv_results

    def evaluate(self, cv_results: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Evaluate model performance.

        Args:
            cv_results: Pre-computed CV results

        Returns:
            DataFrame with performance metrics
        """
        if cv_results is None:
            cv_results = self.cross_validate()

        metrics = performance_metrics(cv_results)

        return metrics

    def get_components(self, forecast: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Extract forecast components.

        Args:
            forecast: Forecast DataFrame

        Returns:
            Dictionary of component DataFrames
        """
        components = {
            'trend': forecast[['ds', 'trend']],
            'yearly': forecast[['ds', 'yearly']] if 'yearly' in forecast else None,
            'weekly': forecast[['ds', 'weekly']] if 'weekly' in forecast else None
        }

        if 'extra_regressors' in forecast:
            components['extra_regressors'] = forecast[['ds', 'extra_regressors']]

        return {k: v for k, v in components.items() if v is not None}

    def add_custom_seasonality(self, name: str, period: float,
                                fourier_order: int) -> None:
        """
        Add custom seasonality.

        Args:
            name: Seasonality name
            period: Period in days
            fourier_order: Fourier order
        """
        if self.model is None:
            raise ValueError("Initialize prophet model first")

        self.model.add_seasonality(name=name, period=period, fourier_order=fourier_order)

    def save(self, filepath: str) -> None:
        """Save the fitted model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Fit the model first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump({
                'params': {
                    'growth': self.growth,
                    'seasonality_mode': self.seasonality_mode,
                    'yearly_seasonality': self.yearly_seasonality,
                    'weekly_seasonality': self.weekly_seasonality,
                    'daily_seasonality': self.daily_seasonality,
                    'changepoint_prior_scale': self.changepoint_prior_scale
                }
            }, f)

        joblib.dump(self.model, filepath.with_suffix('.joblib'))
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> 'ProphetForecaster':
        """Load a fitted model from disk."""
        filepath = Path(filepath)
        self.model = joblib.load(filepath.with_suffix('.joblib'))
        print(f"Model loaded from {filepath}")
        return self
