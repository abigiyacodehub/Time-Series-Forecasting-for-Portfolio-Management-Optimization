"""Tests for portfolio optimization module."""

import pytest
import numpy as np
import pandas as pd
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.efficient_frontier import EfficientFrontier
from src.optimization.risk_metrics import RiskMetrics


@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
    assets = ['AAPL', 'MSFT', 'GOOGL']
    returns = np.random.randn(252, 3) * 0.02  # 2% daily volatility
    return pd.DataFrame(returns, index=dates, columns=assets)


class TestMeanVarianceOptimizer:
    """Tests for MeanVarianceOptimizer."""

    def test_max_sharpe_ratio(self, sample_returns):
        """Test maximum Sharpe ratio optimization."""
        optimizer = MeanVarianceOptimizer(sample_returns)
        result = optimizer.maximize_sharpe_ratio()

        # Check result structure
        assert hasattr(result, 'weights')
        assert hasattr(result, 'expected_return')
        assert hasattr(result, 'expected_volatility')
        assert hasattr(result, 'sharpe_ratio')

        # Check weights sum to 1
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_min_volatility(self, sample_returns):
        """Test minimum volatility optimization."""
        optimizer = MeanVarianceOptimizer(sample_returns)
        result = optimizer.minimize_volatility()

        assert result.expected_volatility > 0
        assert len(result.weights) == len(sample_returns.columns)

    def test_equal_weight(self, sample_returns):
        """Test equal weight portfolio."""
        optimizer = MeanVarianceOptimizer(sample_returns)
        result = optimizer.equal_weight()

        for weight in result.weights.values():
            assert abs(weight - 1/len(sample_returns.columns)) < 0.001


class TestEfficientFrontier:
    """Tests for EfficientFrontier."""

    def test_frontier_calculation(self, sample_returns):
        """Test efficient frontier calculation."""
        optimizer = MeanVarianceOptimizer(sample_returns)
        frontier = EfficientFrontier(
            optimizer.mean_returns.values,
            optimizer.cov_matrix.values,
            optimizer.asset_names
        )

        points = frontier.calculate_frontier(n_points=10)

        assert len(points) > 0
        assert all(p.volatility > 0 for p in points)

    def test_tangency_portfolio(self, sample_returns):
        """Test tangency portfolio calculation."""
        optimizer = MeanVarianceOptimizer(sample_returns)
        frontier = EfficientFrontier(
            optimizer.mean_returns.values,
            optimizer.cov_matrix.values,
            optimizer.asset_names
        )

        frontier.calculate_frontier()
        tangency = frontier.get_tangency_portfolio()

        assert tangency.sharpe_ratio > 0


class TestRiskMetrics:
    """Tests for RiskMetrics."""

    def test_value_at_risk(self, sample_returns):
        """Test VaR calculation."""
        # Use single asset returns
        portfolio_returns = sample_returns.iloc[:, 0]
        risk = RiskMetrics(portfolio_returns, portfolio_value=100000)

        var = risk.value_at_risk(0.95, method='parametric')

        assert var > 0
        assert var < portfolio_returns.std() * 3 * 100000

    def test_max_drawdown(self, sample_returns):
        """Test max drawdown calculation."""
        portfolio_returns = sample_returns.iloc[:, 0]
        risk = RiskMetrics(portfolio_returns)

        max_dd, start, end = risk.max_drawdown()

        assert max_dd >= 0
        assert max_dd <= 1.0

    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        portfolio_returns = sample_returns.iloc[:, 0]
        risk = RiskMetrics(portfolio_returns)

        sharpe = risk.sharpe_ratio()

        assert isinstance(sharpe, float)

    def test_all_metrics(self, sample_returns):
        """Test comprehensive metrics calculation."""
        portfolio_returns = sample_returns.iloc[:, 0]
        risk = RiskMetrics(portfolio_returns)

        metrics = risk.get_all_metrics()

        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) > 0
