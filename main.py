#!/usr/bin/env python3
"""
Time Series Forecasting for Portfolio Management Optimization
Main entry point for the application.
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.data.stock_data import StockDataFetcher
from src.data.preprocessor import DataPreprocessor
from src.models.arima_model import ARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.ensemble_model import EnsembleForecaster
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.efficient_frontier import EfficientFrontier
from src.optimization.risk_metrics import RiskMetrics
from src.utils.backtest import PortfolioBacktest
from src.utils.helpers import save_results, calculate_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Time Series Forecasting for Portfolio Management Optimization'
    )

    parser.add_argument('--tickers', type=str, default='AAPL,MSFT,GOOGL,AMZN,META',
                        help='Comma-separated list of ticker symbols')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--forecast-days', type=int, default=30,
                        help='Number of days to forecast')
    parser.add_argument('--model', type=str, default='arima',
                        choices=['arima', 'prophet', 'lstm', 'ensemble', 'all'],
                        help='Forecasting model to use')
    parser.add_argument('--optimize', action='store_true',
                        help='Run portfolio optimization')
    parser.add_argument('--risk-free-rate', type=float, default=0.02,
                        help='Annual risk-free rate')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to output directory')

    args = parser.parse_args()

    # Set default dates
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')
    start_date = args.start or (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    tickers = [t.strip().upper() for t in args.tickers.split(',')]

    print("=" * 60)
    print("Time Series Forecasting for Portfolio Management Optimization")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Tickers: {tickers}")
    print(f"  Date Range: {start_date} to {end_date}")
    print(f"  Forecast Days: {args.forecast_days}")
    print(f"  Model: {args.model}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Data Acquisition
    print("\n" + "=" * 40)
    print("Step 1: Data Acquisition")
    print("=" * 40)

    fetcher = StockDataFetcher(tickers, start_date, end_date)
    data = fetcher.fetch_data()

    if data.empty:
        print("Error: No data fetched. Exiting.")
        return

    returns = fetcher.calculate_returns()

    # Save summary statistics
    stats = fetcher.get_summary_stats()
    print("\nSummary Statistics:")
    print(stats.to_string())

    stats.to_csv(output_dir / 'summary_statistics.csv')
    print(f"\nSummary statistics saved to {output_dir / 'summary_statistics.csv'}")

    # Step 2: Forecasting
    print("\n" + "=" * 40)
    print("Step 2: Time Series Forecasting")
    print("=" * 40)

    # Select ticker for forecasting (first one by default)
    forecast_ticker = tickers[0]
    series = data[forecast_ticker].dropna()

    forecasts = {}
    forecast_results = {}

    models_to_run = ['arima', 'prophet', 'lstm'] if args.model == 'all' else [args.model]

    for model_name in models_to_run:
        print(f"\n--- {model_name.upper()} Model ---")

        try:
            if model_name == 'arima':
                model = ARIMAForecaster()
                best_order = model.auto_select_order(series, max_p=3, max_q=3)
                model = ARIMAForecaster(order=best_order)
                model.fit(series)
                forecast, confidence = model.predict_with_confidence(steps=args.forecast_days)
                forecasts['ARIMA'] = forecast.values
                forecast_results['ARIMA'] = {
                    'forecast': forecast.values.tolist(),
                    'lower': confidence.iloc[:, 0].values.tolist(),
                    'upper': confidence.iloc[:, 1].values.tolist(),
                    'AIC': model.diagnostics()['AIC']
                }

            elif model_name == 'prophet':
                model = ProphetForecaster()
                model.fit(series)
                forecast_df = model.predict(periods=args.forecast_days)
                forecasts['Prophet'] = forecast_df['yhat'].values
                forecast_results['Prophet'] = {
                    'forecast': forecast_df['yhat'].values.tolist(),
                    'trend': forecast_df['trend'].values.tolist()
                }

            elif model_name == 'lstm':
                model = LSTMForecaster(sequence_length=60, epochs=50)
                model.fit(series, verbose=0)
                forecast = model.predict(steps=args.forecast_days, data=series)
                forecasts['LSTM'] = forecast
                forecast_results['LSTM'] = {
                    'forecast': forecast.tolist()
                }

            elif model_name == 'ensemble':
                # Create ensemble of models
                arima = ARIMAForecaster(order=(1, 1, 1))
                prophet = ProphetForecaster()

                ensemble = EnsembleForecaster([
                    ('ARIMA', arima),
                    ('Prophet', prophet)
                ])

                ensemble.fit(series)
                forecast, individual = ensemble.predict(steps=args.forecast_days, return_individual=True)
                forecasts['Ensemble'] = forecast
                forecast_results['Ensemble'] = {
                    'forecast': forecast.tolist(),
                    'individual': {k: v.tolist() for k, v in individual.items()}
                }

            print(f"Forecast generated for {args.forecast_days} days")

        except Exception as e:
            print(f"Error with {model_name}: {e}")

    # Save forecasts
    with open(output_dir / 'forecasts.json', 'w') as f:
        json.dump(forecast_results, f, indent=2)
    print(f"\nForecasts saved to {output_dir / 'forecasts.json'}")

    # Step 3: Portfolio Optimization
    if args.optimize:
        print("\n" + "=" * 40)
        print("Step 3: Portfolio Optimization")
        print("=" * 40)

        optimizer = MeanVarianceOptimizer(returns, risk_free_rate=args.risk_free_rate)

        # Run different strategies
        results = {}

        print("\n--- Maximum Sharpe Ratio ---")
        max_sharpe = optimizer.maximize_sharpe_ratio()
        results['Max Sharpe'] = max_sharpe
        print(f"Expected Return: {max_sharpe.expected_return:.2%}")
        print(f"Volatility: {max_sharpe.expected_volatility:.2%}")
        print(f"Sharpe Ratio: {max_sharpe.sharpe_ratio:.3f}")

        print("\n--- Minimum Volatility ---")
        min_vol = optimizer.minimize_volatility()
        results['Min Volatility'] = min_vol
        print(f"Expected Return: {min_vol.expected_return:.2%}")
        print(f"Volatility: {min_vol.expected_volatility:.2%}")

        print("\n--- Equal Weight ---")
        equal = optimizer.equal_weight()
        results['Equal Weight'] = equal
        print(f"Expected Return: {equal.expected_return:.2%}")
        print(f"Sharpe Ratio: {equal.sharpe_ratio:.3f}")

        print("\n--- Risk Parity ---")
        risk_parity = optimizer.risk_parity()
        results['Risk Parity'] = risk_parity
        print(f"Expected Return: {risk_parity.expected_return:.2%}")
        print(f"Sharpe Ratio: {risk_parity.sharpe_ratio:.3f}")

        # Efficient Frontier
        print("\n--- Efficient Frontier ---")
        frontier = EfficientFrontier(
            optimizer.mean_returns.values,
            optimizer.cov_matrix.values,
            optimizer.asset_names,
            args.risk_free_rate
        )
        frontier_points = frontier.calculate_frontier(n_points=50)
        frontier_df = frontier.to_dataframe()
        frontier_df.to_csv(output_dir / 'efficient_frontier.csv')
        print(f"Efficient frontier calculated with {len(frontier_points)} points")

        # Save optimization results
        opt_results = []
        for name, result in results.items():
            weights = {k: round(v, 4) for k, v in result.weights.items() if v > 0.01}
            opt_results.append({
                'Strategy': name,
                'Expected Return': f"{result.expected_return:.2%}",
                'Volatility': f"{result.expected_volatility:.2%}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.3f}",
                'Weights': weights
            })

        with open(output_dir / 'optimization_results.json', 'w') as f:
            json.dump(opt_results, f, indent=2)
        print(f"\nOptimization results saved to {output_dir / 'optimization_results.json'}")

    # Step 4: Risk Analysis
    print("\n" + "=" * 40)
    print("Step 4: Risk Analysis")
    print("=" * 40)

    # Use equal-weight portfolio for risk analysis
    portfolio_returns = returns.mean(axis=1)
    risk_metrics = RiskMetrics(portfolio_returns)

    metrics_dict = {
        'Value at Risk (95%)': f"${risk_metrics.value_at_risk(0.95):,.2f}",
        'Expected Shortfall (95%)': f"${risk_metrics.expected_shortfall(0.95):,.2f}",
        'Sharpe Ratio': f"{risk_metrics.sharpe_ratio():.3f}",
        'Sortino Ratio': f"{risk_metrics.sortino_ratio():.3f}",
        'Calmar Ratio': f"{risk_metrics.calmar_ratio():.3f}",
        'Max Drawdown': f"{risk_metrics.max_drawdown()[0]:.2%}",
        'Skewness': f"{risk_metrics.returns.skew():.3f}",
        'Kurtosis': f"{risk_metrics.returns.kurtosis():.3f}"
    }

    print("\nPortfolio Risk Metrics:")
    for metric, value in metrics_dict.items():
        print(f"  {metric}: {value}")

    with open(output_dir / 'risk_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    # Step 5: Generate Report
    print("\n" + "=" * 40)
    print("Step 5: Generating Report")
    print("=" * 40)

    report = f"""
# Portfolio Management Optimization Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Summary
- **Assets:** {', '.join(tickers)}
- **Date Range:** {start_date} to {end_date}
- **Trading Days:** {len(data)}

## Summary Statistics
{stats[['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio']].to_markdown()}

## Forecasts
- **Target:** {forecast_ticker}
- **Forecast Days:** {args.forecast_days}
- **Models Used:** {', '.join(forecasts.keys())}

## Risk Metrics
| Metric | Value |
|--------|-------|
| Value at Risk (95%) | {metrics_dict['Value at Risk (95%)']} |
| Expected Shortfall (95%) | {metrics_dict['Expected Shortfall (95%)']} |
| Sharpe Ratio | {metrics_dict['Sharpe Ratio']} |
| Sortino Ratio | {metrics_dict['Sortino Ratio']} |
| Max Drawdown | {metrics_dict['Max Drawdown']} |
"""

    with open(output_dir / 'report.md', 'w') as f:
        f.write(report)

    print(f"\nReport saved to {output_dir / 'report.md'}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nAll results saved to: {output_dir.absolute()}")
    print("\nTo run the interactive dashboard:")
    print("  streamlit run src/visualization/dashboard.py")


if __name__ == '__main__':
    main()
