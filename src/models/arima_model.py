"""
ARIMA/SARIMA forecasting model for time series prediction.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import Tuple, Optional, Dict, Any
import warnings
import joblib
from pathlib import Path


class ARIMAForecaster:
    """ARIMA-based time series forecaster."""

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize ARIMA forecaster.

        Args:
            order: (p, d, q) order for ARIMA
            seasonal_order: (P, D, Q, s) seasonal order for SARIMA
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model: Optional[Any] = None
        self.fitted_model: Optional[Any] = None
        self._is_seasonal = seasonal_order is not None

    def check_stationarity(self, series: pd.Series, significance: float = 0.05) -> Dict[str, Any]:
        """
        Check stationarity using Augmented Dickey-Fuller test.

        Args:
            series: Time series data
            significance: Significance level for ADF test

        Returns:
            Dictionary with test results
        """
        series = series.dropna()

        result = adfuller(series, autolag='AIC')

        return {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4],
            'Is Stationary': result[1] < significance,
            'Used Lags': result[2],
            'Observations': result[3]
        }

    def auto_select_order(self, series: pd.Series, max_p: int = 5,
                          max_q: int = 5, max_d: int = 2) -> Tuple[int, int, int]:
        """
        Automatically select ARIMA order using AIC.

        Args:
            series: Time series data
            max_p: Maximum AR order
            max_q: Maximum MA order
            max_d: Maximum differencing order

        Returns:
            Optimal (p, d, q) order
        """
        best_aic = np.inf
        best_order = (1, 1, 1)

        series = series.dropna()

        # Determine differencing order
        for d in range(max_d + 1):
            test_series = series.diff(d) if d > 0 else series
            stationarity = self.check_stationarity(test_series.dropna())
            if stationarity['Is Stationary']:
                break

        warnings.filterwarnings('ignore')

        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue

        warnings.filterwarnings('default')
        print(f"Best order: {best_order} with AIC: {best_aic:.2f}")

        return best_order

    def fit(self, data: pd.Series, exogenous: Optional[pd.DataFrame] = None) -> 'ARIMAForecaster':
        """
        Fit ARIMA model to data.

        Args:
            data: Time series to model
            exogenous: Exogenous variables (optional)

        Returns:
            Fitted forecaster
        """
        data = data.dropna()

        if self._is_seasonal:
            self.model = SARIMAX(data, order=self.order,
                                 seasonal_order=self.seasonal_order,
                                 exog=exogenous,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False)
        else:
            self.model = ARIMA(data, order=self.order, exog=exogenous)

        self.fitted_model = self.model.fit()
        print(self.fitted_model.summary())

        return self

    def predict(self, steps: int = 30,
                exogenous: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate forecasts.

        Args:
            steps: Number of steps to forecast
            exogenous: Future exogenous values

        Returns:
            Forecast series
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.fitted_model.forecast(steps=steps, exog=exogenous)

        return forecast

    def predict_with_confidence(self, steps: int = 30,
                                 alpha: float = 0.05) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Generate forecasts with confidence intervals.

        Args:
            steps: Number of steps to forecast
            alpha: Significance level for confidence intervals

        Returns:
            Tuple of (forecast, confidence_intervals)
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)

        return forecast, conf_int

    def get_residuals(self) -> pd.Series:
        """Get model residuals for diagnostic checking."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.fitted_model.resid

    def diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostics.

        Returns:
            Dictionary of diagnostic statistics
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return {
            'AIC': self.fitted_model.aic,
            'BIC': self.fitted_model.bic,
            'Log-Likelihood': self.fitted_model.llf,
            'HQIC': self.fitted_model.hqic,
            'MAE Residuals': np.mean(np.abs(self.fitted_model.resid)),
            'RMSE Residuals': np.sqrt(np.mean(self.fitted_model.resid ** 2))
        }

    def save(self, filepath: str) -> None:
        """Save the fitted model to disk."""
        if self.fitted_model is None:
            raise ValueError("No model to save. Fit the model first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.fitted_model, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> 'ARIMAForecaster':
        """Load a fitted model from disk."""
        self.fitted_model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self
