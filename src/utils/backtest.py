"""
Portfolio backtesting module.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BacktestResult:
    """Backtest result container."""
    portfolio_values: pd.DataFrame
    returns: pd.DataFrame
    metrics: Dict[str, float]
    trades: List[Dict]


class PortfolioBacktest:
    """Portfolio backtesting engine."""

    def __init__(self, initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize backtest.

        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction
            slippage: Slippage model
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def run_backtest(self, prices: pd.DataFrame,
                     weights: Dict[str, float],
                     rebalance_freq: str = 'M') -> BacktestResult:
        """
        Run portfolio backtest.

        Args:
            prices: Historical price data
            weights: Target portfolio weights
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q')

        Returns:
            BacktestResult
        """
        # Initialize
        capital = self.initial_capital
        holdings = {}

        # Get rebalance dates
        returns = prices.pct_change().dropna()
        rebalance_dates = returns.resample(rebalance_freq).first().index

        portfolio_values = []
        portfolio_returns = []
        trades = []

        # Initialize holdings
        for asset in prices.columns:
            if asset in weights:
                shares = (capital * weights[asset]) / prices.loc[prices.index[0], asset]
                holdings[asset] = shares

        prev_value = capital

        for date in prices.index[1:]:
            # Calculate portfolio value
            current_prices = prices.loc[date]
            current_value = sum(holdings.get(asset, 0) * current_prices[asset]
                               for asset in prices.columns)

            # Record portfolio value
            portfolio_values.append({
                'date': date,
                'value': current_value
            })

            # Calculate return
            portfolio_return = (current_value - prev_value) / prev_value
            portfolio_returns.append({
                'date': date,
                'return': portfolio_return
            })

            # Rebalance if needed
            if date in rebalance_dates:
                trade = self._rebalance(current_value, holdings, weights, current_prices, date)
                if trade:
                    trades.append(trade)

            prev_value = current_value

        # Create DataFrames
        values_df = pd.DataFrame(portfolio_values).set_index('date')
        returns_df = pd.DataFrame(portfolio_returns).set_index('date')

        # Calculate metrics
        metrics = self._calculate_metrics(returns_df['return'])

        return BacktestResult(
            portfolio_values=values_df,
            returns=returns_df,
            metrics=metrics,
            trades=trades
        )

    def _rebalance(self, current_value: float, holdings: Dict,
                   target_weights: Dict, prices: pd.Series,
                   date: datetime) -> Optional[Dict]:
        """Rebalance portfolio to target weights."""
        trades = []

        for asset, target_weight in target_weights.items():
            target_value = current_value * target_weight
            target_shares = target_value / prices[asset]
            current_shares = holdings.get(asset, 0)

            trade_size = target_shares - current_shares
            if abs(trade_size) > 1e-6:
                trade_value = abs(trade_size) * prices[asset]
                cost = trade_value * self.transaction_cost

                holdings[asset] = target_shares

                trades.append({
                    'asset': asset,
                    'size': trade_size,
                    'price': prices[asset],
                    'value': trade_value,
                    'cost': cost
                })

        if trades:
            return {
                'date': date,
                'trades': trades,
                'total_cost': sum(t['cost'] for t in trades)
            }

        return None

    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate backtest performance metrics."""
        total_return = (1 + returns).prod() - 1
        annualized_return = returns.mean() * 252
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf

        # Sortino
        downside_std = returns[returns < 0].std() * np.sqrt(252)
        sortino = annualized_return / downside_std if downside_std > 0 else np.inf

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'sortino_ratio': sortino,
            'win_rate': (returns > 0).sum() / len(returns),
            'best_day': returns.max(),
            'worst_day': returns.min()
        }

    def compare_strategies(self, prices: pd.DataFrame,
                           strategies: Dict[str, Dict[str, float]],
                           rebalance_freq: str = 'M') -> pd.DataFrame:
        """
        Compare multiple portfolio strategies.

        Args:
            prices: Historical price data
            strategies: Dictionary of {strategy_name: weights_dict}
            rebalance_freq: Rebalancing frequency

        Returns:
            Comparison DataFrame
        """
        results = {}

        for name, weights in strategies.items():
            backtest = self.run_backtest(prices, weights, rebalance_freq)
            results[name] = backtest.metrics

        return pd.DataFrame(results).T

    def monte_carlo_simulation(self, returns: pd.Series,
                                initial_value: float,
                                n_simulations: int = 1000,
                                horizon: int = 252) -> Dict:
        """
        Run Monte Carlo simulation for portfolio.

        Args:
            returns: Historical returns
            initial_value: Starting portfolio value
            n_simulations: Number of simulations
            horizon: Time horizon in days

        Returns:
            Simulation results
        """
        mean_return = returns.mean()
        std_return = returns.std()

        # Generate random returns
        random_returns = np.random.normal(mean_return, std_return,
                                          (n_simulations, horizon))

        # Calculate paths
        paths = np.zeros((n_simulations, horizon + 1))
        paths[:, 0] = initial_value

        for t in range(horizon):
            paths[:, t + 1] = paths[:, t] * (1 + random_returns[:, t])

        # Calculate statistics
        final_values = paths[:, -1]

        return {
            'paths': paths,
            'mean_path': paths.mean(axis=0),
            'std_path': paths.std(axis=0),
            'final_values': final_values,
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_50': np.percentile(final_values, 50),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'probability_profit': np.mean(final_values > initial_value)
        }
