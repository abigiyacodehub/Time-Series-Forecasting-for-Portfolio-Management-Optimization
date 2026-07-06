"""
Ensemble forecasting model combining multiple models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ForecasterConfig:
    """Configuration for a forecaster."""
    name: str
    weight: float = 1.0
    params: Dict[str, Any] = None


class BaseForecaster(ABC):
    """Abstract base class for forecasters."""

    @abstractmethod
    def fit(self, data: pd.Series, **kwargs) -> 'BaseForecaster':
        pass

    @abstractmethod
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        pass


class EnsembleForecaster:
    """Ensemble of multiple forecasting models."""

    def __init__(self, forecasters: List[Tuple[str, BaseForecaster]],
                 weights: Optional[List[float]] = None):
        """
        Initialize ensemble forecaster.

        Args:
            forecasters: List of (name, forecaster) tuples
            weights: Weights for each forecaster (uniform if None)
        """
        self.forecasters = {name: forecaster for name, forecaster in forecasters}
        self.model_names = [name for name, _ in forecasters]

        if weights is None:
            self.weights = {name: 1.0 / len(forecasters) for name, _ in forecasters}
        else:
            if len(weights) != len(forecasters):
                raise ValueError("Number of weights must match number of forecasters")
            self.weights = {name: weight for name, weight in zip(self.model_names, weights)}

        self.is_fitted: Dict[str, bool] = {name: False for name in self.model_names}
        self.train_data: Optional[pd.Series] = None

    def fit(self, data: pd.Series, **kwargs) -> 'EnsembleForecaster':
        """
        Fit all models in the ensemble.

        Args:
            data: Time series to model
            **kwargs: Additional arguments for individual models

        Returns:
            Fitted ensemble
        """
        self.train_data = data

        for name, forecaster in self.forecasters.items():
            try:
                forecaster.fit(data, **kwargs.get(name, {}))
                self.is_fitted[name] = True
                print(f"Fitted {name} successfully")
            except Exception as e:
                print(f"Warning: Failed to fit {name}: {e}")
                self.is_fitted[name] = False

        return self

    def predict(self, steps: int = 30,
                method: str = 'weighted',
                return_individual: bool = False,
                **kwargs) -> np.ndarray:
        """
        Generate ensemble forecasts.

        Args:
            steps: Number of steps to forecast
            method: 'weighted', 'simple_avg', 'median', 'trimmed_mean'
            return_individual: Return individual predictions
            **kwargs: Additional arguments for individual models

        Returns:
            Ensemble forecast array
        """
        predictions = {}

        for name, forecaster in self.forecasters.items():
            if self.is_fitted[name]:
                try:
                    pred = forecaster.predict(steps=steps, **kwargs.get(name, {}))
                    if isinstance(pred, pd.Series):
                        predictions[name] = pred.values
                    else:
                        predictions[name] = np.array(pred).flatten()
                except Exception as e:
                    print(f"Warning: Failed to predict with {name}: {e}")

        if not predictions:
            raise ValueError("No models available for prediction")

        pred_matrix = np.array(list(predictions.values()))

        if method == 'weighted':
            weights = [self.weights[name] for name in predictions.keys()]
            weights = np.array(weights) / np.sum(weights)
            forecast = np.average(pred_matrix, axis=0, weights=weights)
        elif method == 'simple_avg':
            forecast = np.mean(pred_matrix, axis=0)
        elif method == 'median':
            forecast = np.median(pred_matrix, axis=0)
        elif method == 'trimmed_mean':
            sorted_preds = np.sort(pred_matrix, axis=0)
            n = len(sorted_preds)
            if n > 2:
                forecast = np.mean(sorted_preds[n//4:-n//4], axis=0)
            else:
                forecast = np.mean(sorted_preds, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        if return_individual:
            return forecast, predictions

        return forecast

    def predict_with_confidence(self, steps: int = 30,
                                 alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts with confidence intervals.

        Args:
            steps: Number of steps to forecast
            alpha: Significance level

        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        forecast, predictions = self.predict(steps=steps, return_individual=True)

        pred_matrix = np.array(list(predictions.values()))

        lower = np.percentile(pred_matrix, 100 * alpha / 2, axis=0)
        upper = np.percentile(pred_matrix, 100 * (1 - alpha / 2), axis=0)

        return forecast, lower, upper

    def optimize_weights(self, validation_data: pd.Series,
                         metric: str = 'rmse') -> Dict[str, float]:
        """
        Optimize ensemble weights based on validation performance.

        Args:
            validation_data: Validation period data
            metric: 'rmse', 'mae', or 'mape'

        Returns:
            Optimized weights dictionary
        """
        # Get predictions for validation period
        steps = len(validation_data)
        predictions = {}

        for name, forecaster in self.forecasters.items():
            if self.is_fitted[name]:
                try:
                    pred = forecaster.predict(steps=steps)
                    if isinstance(pred, pd.Series):
                        predictions[name] = pred.values
                    else:
                        predictions[name] = np.array(pred).flatten()
                except:
                    pass

        if not predictions:
            raise ValueError("No models available for optimization")

        # Calculate individual model errors
        actual = validation_data.values
        errors = {}

        for name, pred in predictions.items():
            if metric == 'rmse':
                errors[name] = np.sqrt(np.mean((pred - actual) ** 2))
            elif metric == 'mae':
                errors[name] = np.mean(np.abs(pred - actual))
            elif metric == 'mape':
                errors[name] = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Convert errors to weights (inverse of error)
        total_inverse_error = sum(1.0 / (e + 1e-8) for e in errors.values())

        new_weights = {}
        for name in self.model_names:
            if name in errors:
                new_weights[name] = (1.0 / (errors[name] + 1e-8)) / total_inverse_error
            else:
                new_weights[name] = 0.0

        self.weights = new_weights

        print(f"Optimized weights ({metric}):")
        for name, weight in new_weights.items():
            print(f"  {name}: {weight:.4f}")

        return new_weights

    def get_model_performance(self, validation_data: pd.Series) -> pd.DataFrame:
        """
        Get performance metrics for each model.

        Args:
            validation_data: Validation period data

        Returns:
            DataFrame with performance metrics
        """
        steps = len(validation_data)
        actual = validation_data.values
        metrics = []

        for name, forecaster in self.forecasters.items():
            if self.is_fitted[name]:
                try:
                    pred = forecaster.predict(steps=steps)
                    if isinstance(pred, pd.Series):
                        pred = pred.values
                    else:
                        pred = np.array(pred).flatten()

                    rmse = np.sqrt(np.mean((pred - actual) ** 2))
                    mae = np.mean(np.abs(pred - actual))
                    mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100

                    metrics.append({
                        'Model': name,
                        'Weight': self.weights.get(name, 0),
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE': mape
                    })
                except Exception as e:
                    print(f"Error evaluating {name}: {e}")

        return pd.DataFrame(metrics).sort_values('RMSE')

    def add_forecaster(self, name: str, forecaster: BaseForecaster,
                       weight: Optional[float] = None) -> None:
        """
        Add a new forecaster to the ensemble.

        Args:
            name: Model name
            forecaster: Forecaster instance
            weight: Model weight
        """
        self.forecasters[name] = forecaster
        self.model_names.append(name)
        self.is_fitted[name] = False

        # Re-normalize weights
        if weight is None:
            weight = 1.0 / len(self.model_names)

        self.weights[name] = weight
        total = sum(self.weights.values())
        for model_name in self.weights:
            self.weights[model_name] /= total

    def remove_forecaster(self, name: str) -> None:
        """Remove a forecaster from the ensemble."""
        if name not in self.forecasters:
            raise ValueError(f"Forecaster {name} not found")

        del self.forecasters[name]
        self.model_names.remove(name)
        del self.is_fitted[name]
        del self.weights[name]

        # Re-normalize weights
        if self.weights:
            total = sum(self.weights.values())
            for model_name in self.weights:
                self.weights[model_name] /= total
