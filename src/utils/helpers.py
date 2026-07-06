"""
Utility helper functions.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save results to JSON file.

    Args:
        results: Results dictionary
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy values to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"Results saved to {filepath}")


def calculate_metrics(returns: pd.Series, positions: pd.Series = None) -> Dict[str, float]:
    """
    Calculate trading performance metrics.

    Args:
        returns: Strategy returns
        positions: Optional position sizes

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = returns.mean() * 252
    metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility'] if metrics['annualized_volatility'] > 0 else 0

    # Drawdown metrics
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()

    # Win/loss metrics
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    metrics['win_rate'] = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    metrics['avg_win'] = positive_returns.mean() if len(positive_returns) > 0 else 0
    metrics['avg_loss'] = negative_returns.mean() if len(negative_returns) > 0 else 0
    metrics['profit_factor'] = positive_returns.sum() / abs(negative_returns.sum()) if negative_returns.sum() != 0 else np.inf

    # Additional metrics
    metrics['skewness'] = returns.skew()
    metrics['kurtosis'] = returns.kurtosis()
    metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else np.inf

    # Sortino ratio
    downside_std = returns[returns < 0].std() * np.sqrt(252)
    metrics['sortino_ratio'] = metrics['annualized_return'] / downside_std if downside_std > 0 else np.inf

    return metrics


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value:.2%}"


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def create_summary_table(metrics_list: list, index_names: list = None) -> pd.DataFrame:
    """
    Create summary comparison table.

    Args:
        metrics_list: List of metrics dictionaries
        index_names: Optional names for each row

    Returns:
        Comparison DataFrame
    """
    df = pd.DataFrame(metrics_list)

    if index_names:
        df.index = index_names

    return df


def get_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Get trading days between dates.

    Args:
        start_date: Start date string
        end_date: End date string

    Returns:
        DatetimeIndex of trading days
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    return dates


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize returns.

    Args:
        returns: Returns series
        periods_per_year: Number of periods in a year

    Returns:
        Annualized return
    """
    return (1 + returns.mean()) ** periods_per_year - 1


def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize volatility.

    Args:
        returns: Returns series
        periods_per_year: Number of periods in a year

    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Check data quality and report issues.

    Args:
        data: DataFrame to check

    Returns:
        Data quality report
    """
    report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'total_missing': data.isnull().sum().sum(),
        'missing_percentage': data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100,
        'duplicate_rows': data.duplicated().sum(),
        'date_range': None,
        'issues': []
    }

    if isinstance(data.index, pd.DatetimeIndex):
        report['date_range'] = f"{data.index.min()} to {data.index.max()}"

    if report['missing_percentage'] > 10:
        report['issues'].append(f"High missing value percentage: {report['missing_percentage']:.1f}%")

    if report['duplicate_rows'] > 0:
        report['issues'].append(f"Found {report['duplicate_rows']} duplicate rows")

    return report
