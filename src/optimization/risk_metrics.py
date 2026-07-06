"""
Risk metrics and portfolio analysis tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats


class RiskMetrics:
    """Calculate various risk metrics for portfolio analysis."""

    def __init__(self, returns: pd.Series, portfolio_value: float = 100000):
        """
        Initialize risk metrics calculator.

        Args:
            returns: Portfolio returns series
            portfolio_value: Current portfolio value
        """
        self.returns = returns.dropna()
        self.portfolio_value = portfolio_value

    def value_at_risk(self, confidence: float = 0.95,
                      method: str = 'parametric') -> float:
        """
        Calculate Value at Risk.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)
            method: 'parametric', 'historical', or 'monte_carlo'

        Returns:
            VaR value (positive number representing potential loss)
        """
        if method == 'parametric':
            mu = self.returns.mean()
            sigma = self.returns.std()
            z_score = stats.norm.ppf(1 - confidence)
            var = -(mu + z_score * sigma) * self.portfolio_value
        elif method == 'historical':
            var = -np.percentile(self.returns, (1 - confidence) * 100) * self.portfolio_value
        elif method == 'monte_carlo':
            mu = self.returns.mean()
            sigma = self.returns.std()
            simulated = np.random.normal(mu, sigma, 10000)
            var = -np.percentile(simulated, (1 - confidence) * 100) * self.portfolio_value
        else:
            raise ValueError(f"Unknown method: {method}")

        return max(var, 0)

    def expected_shortfall(self, confidence: float = 0.95,
                           method: str = 'parametric') -> float:
        """
        Calculate Expected Shortfall (CVaR).

        Args:
            confidence: Confidence level
            method: 'parametric' or 'historical'

        Returns:
            ES value (positive number representing potential loss)
        """
        if method == 'historical':
            var = -np.percentile(self.returns, (1 - confidence) * 100)
            es = -self.returns[self.returns <= -var].mean() * self.portfolio_value
        elif method == 'parametric':
            mu = self.returns.mean()
            sigma = self.returns.std()
            z_score = stats.norm.ppf(1 - confidence)

            # Expected Shortfall formula for normal distribution
            es_factor = stats.norm.pdf(z_score) / (1 - confidence)
            es = (sigma * es_factor - mu) * self.portfolio_value
        else:
            raise ValueError(f"Unknown method: {method}")

        return max(es, 0)

    def max_drawdown(self) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown.

        Returns:
            Tuple of (max_dd, start_date, end_date)
        """
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max

        max_dd = drawdown.min()
        end_idx = drawdown.idxmin()

        # Find start of drawdown
        cumulative_until = drawdown.loc[:end_idx]
        start_idx = cumulative_until[cumulative_until == 0].last_valid_index() or drawdown.index[0]

        return abs(max_dd), start_idx, end_idx

    def calmar_ratio(self, periods: int = 3) -> float:
        """
        Calculate Calmar Ratio.

        Args:
            periods: Number of years for annualized return

        Returns:
            Calmar ratio
        """
        annualized_return = self.returns.mean() * 252
        max_dd, _, _ = self.max_drawdown()

        if max_dd == 0:
            return np.inf

        return annualized_return / max_dd

    def sortino_ratio(self, target_return: float = 0.0) -> float:
        """
        Calculate Sortino Ratio.

        Args:
            target_return: Target return for downside calculation

        Returns:
            Sortino ratio
        """
        annualized_return = self.returns.mean() * 252
        downside_returns = self.returns[self.returns < target_return]

        if len(downside_returns) == 0:
            return np.inf

        downside_std = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)

        return annualized_return / downside_std if downside_std > 0 else np.inf

    def treynor_ratio(self, beta: float, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Treynor Ratio.

        Args:
            beta: Portfolio beta
            risk_free_rate: Annual risk-free rate

        Returns:
            Treynor ratio
        """
        annualized_return = self.returns.mean() * 252
        return (annualized_return - risk_free_rate) / beta if beta != 0 else 0

    def information_ratio(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio.

        Args:
            benchmark_returns: Benchmark returns series

        Returns:
            Information ratio
        """
        excess_returns = self.returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        active_return = excess_returns.mean() * 252

        return active_return / tracking_error if tracking_error > 0 else 0

    def beta(self, market_returns: pd.Series) -> float:
        """
        Calculate portfolio beta.

        Args:
            market_returns: Market index returns

        Returns:
            Beta value
        """
        covariance = self.returns.cov(market_returns)
        market_variance = market_returns.var()

        return covariance / market_variance if market_variance > 0 else 0

    def alpha(self, market_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Jensen's Alpha.

        Args:
            market_returns: Market index returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Alpha value (annualized)
        """
        port_return = self.returns.mean() * 252
        market_return = market_returns.mean() * 252
        port_beta = self.beta(market_returns)

        alpha = port_return - (risk_free_rate + port_beta * (market_return - risk_free_rate))
        return alpha

    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe Ratio.

        Args:
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        annualized_return = self.returns.mean() * 252
        annualized_vol = self.returns.std() * np.sqrt(252)

        return (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0

    def rolling_sharpe(self, window: int = 252, risk_free_rate: float = 0.02) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.

        Args:
            window: Rolling window size
            risk_free_rate: Annual risk-free rate

        Returns:
            Rolling Sharpe ratio series
        """
        rolling_return = self.returns.rolling(window).mean() * 252
        rolling_std = self.returns.rolling(window).std() * np.sqrt(252)

        return (rolling_return - risk_free_rate) / rolling_std

    def tail_ratio(self) -> float:
        """
        Calculate Tail Ratio.

        Returns:
            Tail ratio
        """
        returns = self.returns.values
        tenth_percentile = np.percentile(returns, 10)
        ninetieth_percentile = np.percentile(returns, 90)

        if abs(tenth_percentile) < 1e-10:
            return np.inf

        return abs(ninetieth_percentile / tenth_percentile)

    def get_all_metrics(self, market_returns: Optional[pd.Series] = None,
                        risk_free_rate: float = 0.02) -> pd.DataFrame:
        """
        Calculate all risk metrics.

        Args:
            market_returns: Optional market returns for beta/alpha
            risk_free_rate: Annual risk-free rate

        Returns:
            DataFrame with all metrics
        """
        metrics = {
            'Annualized Return': self.returns.mean() * 252,
            'Annualized Volatility': self.returns.std() * np.sqrt(252),
            'Sharpe Ratio': self.sharpe_ratio(risk_free_rate),
            'Sortino Ratio': self.sortino_ratio(),
            'Calmar Ratio': self.calmar_ratio(),
            'VaR (95%)': self.value_at_risk(0.95),
            'Expected Shortfall (95%)': self.expected_shortfall(0.95),
            'Tail Ratio': self.tail_ratio()
        }

        max_dd, start, end = self.max_drawdown()
        metrics['Max Drawdown'] = max_dd
        metrics['Max DD Start'] = start
        metrics['Max DD End'] = end

        if market_returns is not None:
            metrics['Beta'] = self.beta(market_returns)
            metrics['Alpha'] = self.alpha(market_returns, risk_free_rate)
            metrics['Treynor Ratio'] = self.treynor_ratio(metrics['Beta'], risk_free_rate)
            metrics['Information Ratio'] = self.information_ratio(market_returns)

        # Skewness and Kurtosis
        metrics['Skewness'] = self.returns.skew()
        metrics['Kurtosis'] = self.returns.kurtosis()

        return pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
