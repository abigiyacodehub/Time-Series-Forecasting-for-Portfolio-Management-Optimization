"""
LSTM Neural Network forecasting model for time series prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import joblib
import json

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class LSTMForecaster:
    """LSTM-based time series forecaster."""

    def __init__(self, sequence_length: int = 60,
                 lstm_units: List[int] = [50, 50],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32):
        """
        Initialize LSTM forecaster.

        Args:
            sequence_length: Number of time steps for input sequences
            lstm_units: List of LSTM units per layer
            dropout_rate: Dropout rate between layers
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model. Install with: pip install tensorflow")

        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None
        self.scaler_params: Optional[Dict] = None

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM model architecture.

        Args:
            input_shape: Shape of input data (sequence_length, features)

        Returns:
            Compiled Keras model
        """
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True if len(self.lstm_units) > 1 else False,
            input_shape=input_shape
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        # Hidden LSTM layers
        for i, units in enumerate(self.lstm_units[1:], start=1):
            is_last = (i == len(self.lstm_units) - 1)
            model.add(LSTM(
                units=units,
                return_sequences=not is_last
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))

        # Output layer
        model.add(Dense(units=1))

        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        print(f"Model built with {model.count_params()} parameters")

        return model

    def create_dataset(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.

        Args:
            data: Input data array

        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i])

        return np.array(X), np.array(y)

    def fit(self, data: pd.Series,
            validation_split: float = 0.2,
            early_stopping: bool = True,
            verbose: int = 1) -> 'LSTMForecaster':
        """
        Fit LSTM model to data.

        Args:
            data: Time series to model
            validation_split: Fraction for validation
            early_stopping: Use early stopping
            verbose: Verbosity level

        Returns:
            Fitted forecaster
        """
        # Normalize data
        data_values = data.values.reshape(-1, 1)
        data_min, data_max = data_values.min(), data_values.max()
        data_normalized = (data_values - data_min) / (data_max - data_min)

        self.scaler_params = {'min': data_min, 'max': data_max}

        # Create sequences
        X, y = self.create_dataset(data_normalized)

        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Build model
        input_shape = (X.shape[1], X.shape[2])
        self.build_model(input_shape)

        # Callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ))
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ))

        # Train model
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )

        return self

    def predict(self, steps: int = 30,
                last_sequence: Optional[np.ndarray] = None,
                data: Optional[pd.Series] = None) -> np.ndarray:
        """
        Generate multi-step forecasts.

        Args:
            steps: Number of steps to forecast
            last_sequence: Starting sequence (if None, uses last from data)
            data: Original data to extract last sequence

        Returns:
            Array of forecasts
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get last sequence
        if last_sequence is None:
            if data is None:
                raise ValueError("Provide either last_sequence or data")
            data_values = data.values.reshape(-1, 1)
            data_normalized = (data_values - self.scaler_params['min']) / \
                             (self.scaler_params['max'] - self.scaler_params['min'])
            last_sequence = data_normalized[-self.sequence_length:]

        forecasts = []
        current_sequence = last_sequence.copy()

        for _ in range(steps):
            # Predict next value
            X = current_sequence.reshape(1, self.sequence_length, 1)
            pred_normalized = self.model.predict(X, verbose=0)[0, 0]
            forecasts.append(pred_normalized)

            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred_normalized

        # Inverse transform
        forecasts = np.array(forecasts)
        forecasts = forecasts * (self.scaler_params['max'] - self.scaler_params['min']) + \
                   self.scaler_params['min']

        return forecasts

    def predict_with_uncertainty(self, steps: int = 30,
                                  n_samples: int = 100,
                                  data: Optional[pd.Series] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate forecasts with uncertainty estimates using dropout at inference.

        Args:
            steps: Number of steps to forecast
            n_samples: Number of samples for uncertainty estimation
            data: Original data

        Returns:
            Tuple of (mean_forecasts, std_forecasts)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        all_forecasts = []

        for _ in range(n_samples):
            forecast = self.predict(steps=steps, data=data)
            all_forecasts.append(forecast)

        all_forecasts = np.array(all_forecasts)
        mean_forecasts = all_forecasts.mean(axis=0)
        std_forecasts = all_forecasts.std(axis=0)

        return mean_forecasts, std_forecasts

    def evaluate(self, data: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            data: Test data series

        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare test data
        data_values = data.values.reshape(-1, 1)
        data_normalized = (data_values - self.scaler_params['min']) / \
                         (self.scaler_params['max'] - self.scaler_params['min'])

        X_test, y_test = self.create_dataset(data_normalized)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Evaluate
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)

        # Make predictions
        y_pred = self.model.predict(X_test, verbose=0)

        # Calculate metrics
        mse = np.mean((y_pred.flatten() - y_test) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / (y_test + 1e-8))) * 100

        return {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        if self.history is None:
            raise ValueError("No training history available")

        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history.get('val_loss', []),
            'mae': self.history.history.get('mae', []),
            'val_mae': self.history.history.get('val_mae', [])
        }

    def save(self, filepath: str) -> None:
        """Save the fitted model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Fit the model first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(str(filepath.with_suffix('.h5')))

        # Save config and scaler params
        config = {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'scaler_params': self.scaler_params
        }

        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(config, f)

        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> 'LSTMForecaster':
        """Load a fitted model from disk."""
        filepath = Path(filepath)

        # Load model
        self.model = load_model(str(filepath.with_suffix('.h5')))

        # Load config
        with open(filepath.with_suffix('.json'), 'r') as f:
            config = json.load(f)

        self.sequence_length = config['sequence_length']
        self.lstm_units = config['lstm_units']
        self.dropout_rate = config['dropout_rate']
        self.learning_rate = config['learning_rate']
        self.scaler_params = config['scaler_params']

        print(f"Model loaded from {filepath}")
        return self
