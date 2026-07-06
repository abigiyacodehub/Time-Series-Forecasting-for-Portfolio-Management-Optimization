"""
Mean-Variance Portfolio Optimization (Markowitz Model).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import cvxpy as cp


@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str


class MeanVarianceOptimizer:
    """
    Mean-Variance Portfolio Optimizer based on Modern Portfolio Theory.
    """

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize optimizer.

        Args:
            returns: DataFrame of asset returns
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        self.asset_names = list(returns.columns)
        self.n_assets = len(self.asset_names)

    def maximize_sharpe_ratio(self, allow_short: bool = False,
                               max_weight: Optional[float] = None,
                               min_weight: float = 0.0) -> OptimizationResult:
        """
        Find portfolio with maximum Sharpe ratio.

        Args:
            allow_short: Allow short selling
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset

        Returns:
            OptimizationResult
        """
        n = self.n_assets
        mu = self.mean_returns.values
        sigma = self.cov_matrix.values

        # Optimization variables
        w = cp.Variable(n)
        ret = mu.T @ w
        risk = cp.quad_form(w, sigma)

        # Objective: maximize Sharpe ratio
        # Sharpe = (ret - rf) / sqrt(risk)
        # Use Charnes-Cooper transformation
        y = cp.Variable()
        x = cp.Variable(n)

        constraints = [
            x >= 0 if not allow_short else x >= y * min_weight,
            y * self.risk_free_rate + mu.T @ x >= 1.0,
            cp.quad_form(x, sigma) <= 1.0,
            y >= 0
        ]

        if max_weight:
            constraints.append(x <= y * max_weight)

        # Sum constraint
        constraints.append(cp.sum(x) == y)

        # Solve for y first (minimize variance for unit expected excess return)
        objective = cp.Minimize(cp.quad_form(x, sigma))

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if problem.status != 'optimal':
            print(f"Warning: Optimization status {problem.status}")

        # Convert back to weights
        weights = x.value / y.value if y.value > 0 else np.ones(n) / n

        # Calculate metrics
        weights_dict = dict(zip(self.asset_names, weights))
        expected_return = mu.T @ weights
        expected_vol = np.sqrt(weights.T @ sigma @ weights)
        sharpe = (expected_return - self.risk_free_rate) / expected_vol

        return OptimizationResult(
            weights=weights_dict,
            expected_return=expected_return,
            expected_volatility=expected_vol,
            sharpe_ratio=sharpe,
            method='max_sharpe'
        )

    def minimize_volatility(self, target_return: Optional[float] = None,
                            allow_short: bool = False,
                            max_weight: Optional[float] = None) -> OptimizationResult:
        """
        Find minimum volatility portfolio.

        Args:
            target_return: Target annual return (if provided)
            allow_short: Allow short selling
            max_weight: Maximum weight per asset

        Returns:
            OptimizationResult
        """
        n = self.n_assets
        mu = self.mean_returns.values
        sigma = self.cov_matrix.values

        w = cp.Variable(n)
        risk = cp.quad_form(w, sigma)

        constraints = [
            cp.sum(w) == 1,
            w >= 0 if not allow_short else w >= -1
        ]

        if target_return:
            constraints.append(mu.T @ w >= target_return)

        if max_weight:
            constraints.append(w <= max_weight)

        objective = cp.Minimize(risk)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if problem.status != 'optimal':
            print(f"Warning: Optimization status {problem.status}")

        weights = w.value if w.value is not None else np.ones(n) / n

        weights_dict = dict(zip(self.asset_names, weights))
        expected_return = mu.T @ weights
        expected_vol = np.sqrt(weights.T @ sigma @ weights)
        sharpe = (expected_return - self.risk_free_rate) / expected_vol

        return OptimizationResult(
            weights=weights_dict,
            expected_return=expected_return,
            expected_volatility=expected_vol,
            sharpe_ratio=sharpe,
            method='min_volatility'
        )

    def maximize_return(self, target_volatility: float,
                        allow_short: bool = False,
                        max_weight: Optional[float] = None) -> OptimizationResult:
        """
        Find maximum return portfolio for a given risk level.

        Args:
            target_volatility: Target annual volatility
            allow_short: Allow short selling
            max_weight: Maximum weight per asset

        Returns:
            OptimizationResult
        """
        n = self.n_assets
        mu = self.mean_returns.values
        sigma = self.cov_matrix.values

        w = cp.Variable(n)
        ret = mu.T @ w
        risk = cp.quad_form(w, sigma)

        constraints = [
            cp.sum(w) == 1,
            w >= 0 if not allow_short else w >= -1,
            risk <= target_volatility ** 2
        ]

        if max_weight:
            constraints.append(w <= max_weight)

        objective = cp.Maximize(ret)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if problem.status != 'optimal':
            print(f"Warning: Optimization status {problem.status}")

        weights = w.value if w.value is not None else np.ones(n) / n

        weights_dict = dict(zip(self.asset_names, weights))
        expected_return = mu.T @ weights
        expected_vol = np.sqrt(weights.T @ sigma @ weights)
        sharpe = (expected_return - self.risk_free_rate) / expected_vol

        return OptimizationResult(
            weights=weights_dict,
            expected_return=expected_return,
            expected_volatility=expected_vol,
            sharpe_ratio=sharpe,
            method='max_return'
        )

    def equal_weight(self) -> OptimizationResult:
        """Equal-weight portfolio."""
        n = self.n_assets
        weights = np.ones(n) / n

        weights_dict = dict(zip(self.asset_names, weights))
        mu = self.mean_returns.values
        sigma = self.cov_matrix.values

        expected_return = mu.T @ weights
        expected_vol = np.sqrt(weights.T @ sigma @ weights)
        sharpe = (expected_return - self.risk_free_rate) / expected_vol

        return OptimizationResult(
            weights=weights_dict,
            expected_return=expected_return,
            expected_volatility=expected_vol,
            sharpe_ratio=sharpe,
            method='equal_weight'
        )

    def risk_parity(self, target_risk_budget: Optional[np.ndarray] = None) -> OptimizationResult:
        """
        Risk parity portfolio optimization.

        Args:
            target_risk_budget: Target risk contribution per asset (equal if None)

        Returns:
            OptimizationResult
        """
        n = self.n_assets
        sigma = self.cov_matrix.values

        if target_risk_budget is None:
            target_risk_budget = np.ones(n) / n

        # Use iterative approach for risk parity
        weights = np.ones(n) / n

        for _ in range(100):
            # Calculate marginal risk contribution
            portfolio_vol = np.sqrt(weights.T @ sigma @ weights)
            marginal_contrib = (sigma @ weights) / portfolio_vol

            # Risk contribution
            risk_contrib = weights * marginal_contrib / portfolio_vol

            # Adjust weights
            adj_factor = target_risk_budget / (risk_contrib + 1e-10)
            weights = weights * np.sqrt(adj_factor)
            weights = weights / np.sum(weights)

        weights_dict = dict(zip(self.asset_names, weights))
        mu = self.mean_returns.values
        expected_return = mu.T @ weights
        expected_vol = np.sqrt(weights.T @ sigma @ weights)
        sharpe = (expected_return - self.risk_free_rate) / expected_vol

        return OptimizationResult(
            weights=weights_dict,
            expected_return=expected_return,
            expected_volatility=expected_vol,
            sharpe_ratio=sharpe,
            method='risk_parity'
        )

    def black_litterman(self, views: Dict[str, float],
                        view_uncertainty: Optional[Dict[str, float]] = None,
                        tau: float = 0.05) -> OptimizationResult:
        """
        Black-Litterman portfolio optimization.

        Args:
            views: Dictionary of asset views (expected returns)
            view_uncertainty: Confidence in each view
            tau: Scaling factor for prior

        Returns:
            OptimizationResult
        """
        n = self.n_assets
        mu = self.mean_returns.values
        sigma = self.cov_matrix.values

        # Market implied returns (reverse optimization)
        market_weights = self.equal_weight().weights
        market_weights = np.array([market_weights[name] for name in self.asset_names])
        market_implied_returns = tau * sigma @ market_weights

        # Incorporate views
        if views:
            P = np.zeros((len(views), n))
            Q = np.zeros(len(views))

            for i, (asset, view) in enumerate(views.items()):
                if asset in self.asset_names:
                    P[i, self.asset_names.index(asset)] = 1
                    Q[i] = view

            # View uncertainty matrix
            if view_uncertainty:
                omega = np.diag([view_uncertainty.get(name, 0.01) for name in views.keys()])
            else:
                omega = np.eye(len(views)) * 0.01

            # Black-Litterman expected returns
            M = np.linalg.inv(np.linalg.inv(tau * sigma) + P.T @ np.linalg.inv(omega) @ P)
            posterior_returns = M @ (np.linalg.inv(tau * sigma) @ market_implied_returns +
                                     P.T @ np.linalg.inv(omega) @ Q)

            posterior_cov = sigma + M
        else:
            posterior_returns = market_implied_returns
            posterior_cov = sigma

        # Maximize Sharpe using posterior estimates
        w = cp.Variable(n)
        ret = posterior_returns.T @ w
        risk = cp.quad_form(w, posterior_cov)

        constraints = [cp.sum(w) == 1, w >= 0]
        objective = cp.Minimize(risk - ret)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        weights = w.value if w.value is not None else np.ones(n) / n

        weights_dict = dict(zip(self.asset_names, weights))
        expected_return = posterior_returns.T @ weights
        expected_vol = np.sqrt(weights.T @ posterior_cov @ weights)
        sharpe = (expected_return - self.risk_free_rate) / expected_vol

        return OptimizationResult(
            weights=weights_dict,
            expected_return=expected_return,
            expected_volatility=expected_vol,
            sharpe_ratio=sharpe,
            method='black_litterman'
        )

    def get_portfolio_summary(self, result: OptimizationResult) -> pd.DataFrame:
        """
        Get summary DataFrame for optimization result.

        Args:
            result: OptimizationResult

        Returns:
            Summary DataFrame
        """
        weights = [result.weights[name] for name in self.asset_names]

        df = pd.DataFrame({
            'Asset': self.asset_names,
            'Weight': weights,
            'Expected Return': self.mean_returns.values * weights,
            'Volatility Contribution': np.sqrt(np.diag(self.cov_matrix)) * np.abs(weights)
        })

        df = df.sort_values('Weight', ascending=False)

        summary = pd.DataFrame([{
            'Method': result.method,
            'Expected Annual Return': f"{result.expected_return:.2%}",
            'Expected Annual Volatility': f"{result.expected_volatility:.2%}",
            'Sharpe Ratio': f"{result.sharpe_ratio:.3f}",
            'Number of Assets': np.sum(np.array(weights) > 0.01)
        }])

        return df, summary
