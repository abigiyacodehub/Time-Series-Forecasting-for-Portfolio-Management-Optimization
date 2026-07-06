"""
Run ARIMA forecasting for portfolio optimization.
Usage: python scripts/run_arima.py --ticker SPY --days 30
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.arima_model import ARIMAForecaster
from src.data.preprocessor import DataPreprocessor


def main():
    parser = argparse.ArgumentParser(description='Run ARIMA forecasting')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker to forecast')
    parser.add_argument('--days', type=int, default=30, help='Forecast days')
    parser.add_argument('--data-path', type=str, default='data/raw/stock_prices.csv', help='Path to price data')
    parser.add_argument('--output-path', type=str, default='data/processed/', help='Output directory')

    args = parser.parse_args()

    print("=" * 60)
    print(f"ARIMA Forecast for {args.ticker}")
    print("=" * 60)

    # Load data
    try:
        prices = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
        if args.ticker not in prices.columns:
            raise ValueError(f"Ticker {args.ticker} not found in data")

        series = prices[args.ticker].dropna()
        print(f"\nLoaded {len(series)} observations")
        print(f"Date range: {series.index.min()} to {series.index.max()}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data_path}")
        print("Please run fetch_data.py first")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Train/test split (chronological)
    test_size = 0.2
    split_idx = int(len(series) * (1 - test_size))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    print(f"\nTrain set: {len(train)} observations")
    print(f"Test set: {len(test)} observations")

    # Fit ARIMA model
    print(f"\nFitting ARIMA model...")

    try:
        # Auto-select order
        forecaster = ARIMAForecaster()
        best_order = forecaster.auto_select_order(train, max_p=5, max_q=5)

        print(f"\nBest ARIMA order: {best_order}")

        # Fit final model
        forecaster = ARIMAForecaster(order=best_order)
        forecaster.fit(train)

        print("\nModel Summary:")
        print(forecaster.fitted_model.summary())

        # Generate forecast
        forecast, conf_int = forecaster.predict_with_confidence(
            steps=min(args.days, len(test)),
            alpha=0.05
        )

        # Set proper index
        forecast_dates = pd.date_range(start=test.index[0], periods=len(forecast), freq='B')

        # Calculate metrics
        actual_test = test.iloc[:len(forecast)]
        rmse = np.sqrt(np.mean((forecast.values - actual_test.values) ** 2))
        mape = np.mean(np.abs((actual_test.values - forecast.values) / actual_test.values)) * 100

        print(f"\nForecast Metrics:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")

        # Save results
        os.makedirs(args.output_path, exist_ok=True)

        results = pd.DataFrame({
            'Date': forecast_dates,
            'Actual': actual_test.values,
            'Forecast': forecast.values,
            'Lower_95': conf_int.iloc[:, 0].values,
            'Upper_95': conf_int.iloc[:, 1].values
        })

        output_file = os.path.join(args.output_path, f'{args.ticker}_arima_forecast.csv')
        results.to_csv(output_file, index=False)
        print(f"\nForecast saved to: {output_file}")

    except Exception as e:
        print(f"Error during forecasting: {e}")
        raise


if __name__ == '__main__':
    main()
