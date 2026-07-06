"""
Efficient Frontier calculation and visualization.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import cvxpy as cp


@dataclass
class FrontierPoint:
    """Single point on the efficient frontier."""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: Dict[str, float]


class EfficientFrontier:
    """
    Efficient Frontier calculator and analyzer.
    """

    def __init__(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                 asset_names: List[str], risk_free_rate: float = 0.02):
        """
        Initialize efficient frontier.

        Args:
            mean_returns: Annualized mean returns
            cov_matrix: Annualized covariance matrix
            asset_names: List of asset names
            risk_free_rate: Annual risk-free rate
        """
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.asset_names = asset_names
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(asset_names)
        self.frontier_points: List[FrontierPoint] = []

    def calculate_frontier(self, n_points: int = 100,
                           allow_short: bool = False,
                           max_weight: Optional[float] = None) -> List[FrontierPoint]:
        """
        Calculate the efficient frontier.

        Args:
            n_points: Number of points to calculate
            allow_short: Allow short positions
            max_weight: Maximum weight per asset

        Returns:
            List of FrontierPoint objects
        """
        min_ret = np.min(self.mean_returns)
        max_ret = np.max(self.mean_returns)
        target_returns = np.linspace(min_ret, max_ret, n_points)

        self.frontier_points = []

        for target_ret in target_returns:
            result = self._optimize_for_target_return(
                target_ret, allow_short, max_weight
            )

            if result is not None:
                self.frontier_points.append(result)

        return self.frontier_points

    def _optimize_for_target_return(self, target_return: float,
                                    allow_short: bool,
                                    max_weight: Optional[float]) -> Optional[FrontierPoint]:
        """Optimize portfolio for a target return."""
        n = self.n_assets
        mu = self.mean_returns
        sigma = self.cov_matrix

        w = cp.Variable(n)
        portfolio_ret = mu.T @ w
        portfolio_var = cp.quad_form(w, sigma)

        constraints = [
            cp.sum(w) == 1,
            portfolio_ret >= target_return
        ]

        if allow_short:
            constraints.append(w >= -1)
        else:
            constraints.append(w >= 0)

        if max_weight:
            constraints.append(w <= max_weight)

        objective = cp.Minimize(portfolio_var)

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.SCS)

            if problem.status == 'optimal' and w.value is not None:
                weights = w.value
                weights_dict = dict(zip(self.asset_names, weights))
                expected_ret = mu.T @ weights
                expected_vol = np.sqrt(weights.T @ sigma @ weights)
                sharpe = (expected_ret - self.risk_free_rate) / expected_vol

                return FrontierPoint(
                    expected_return=expected_ret,
                    volatility=expected_vol,
                    sharpe_ratio=sharpe,
                    weights=weights_dict
                )
        except:
            pass

        return None

    def get_optimal_portfolio(self, criterion: str = 'max_sharpe') -> FrontierPoint:
        """
        Get optimal portfolio based on criterion.

        Args:
            criterion: 'max_sharpe', 'min_volatility', or 'max_return'

        Returns:
            Optimal FrontierPoint
        """
        if not self.frontier_points:
            self.calculate_frontier()

        if criterion == 'max_sharpe':
            return max(self.frontier_points, key=lambda x: x.sharpe_ratio)
        elif criterion == 'min_volatility':
            return min(self.frontier_points, key=lambda x: x.volatility)
        elif criterion == 'max_return':
            return max(self.frontier_points, key=lambda x: x.expected_return)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def get_portfolio_at_risk_level(self, target_volatility: float) -> Optional[FrontierPoint]:
        """
        Get portfolio at a specific risk level.

        Args:
            target_volatility: Target volatility

        Returns:
            FrontierPoint or None
        """
        if not self.frontier_points:
            self.calculate_frontier()

        # Find closest point
        for point in sorted(self.frontier_points, key=lambda x: x.volatility):
            if point.volatility >= target_volatility:
                return point

        return None

    def get_tangency_portfolio(self) -> FrontierPoint:
        """Get the tangency portfolio (maximum Sharpe ratio)."""
        return self.get_optimal_portfolio('max_sharpe')

    def calculate_cal(self, n_points: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Capital Allocation Line.

        Args:
            n_points: Number of points

        Returns:
            Tuple of (volatilities, returns) arrays
        """
        # Get max Sharpe portfolio
        tangency = self.get_tangency_portfolio()

        # CAL: E(r) = rf + (Sharpe * sigma)
        max_vol = tangency.volatility * 2
        volatilities = np.linspace(0, max_vol, n_points)
        returns = self.risk_free_rate + tangency.sharpe_ratio * volatilities

        return volatilities, returns

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert frontier points to DataFrame.

        Returns:
            DataFrame with frontier points
        """
        if not self.frontier_points:
            self.calculate_frontier()

        data = []

        for point in self.frontier_points:
            row = {
                'Expected Return': point.expected_return,
                'Volatility': point.volatility,
                'Sharpe Ratio': point.sharpe_ratio
            }

            for asset, weight in point.weights.items():
                row[f'Weight_{asset}'] = weight

            data.append(row)

        return pd.DataFrame(data)

    def get_risk_contribution(self, point: FrontierPoint) -> Dict[str, float]:
        """
        Calculate risk contribution of each asset.

        Args:
            point: FrontierPoint

        Returns:
            Dictionary of risk contributions
        """
        weights = np.array([point.weights[name] for name in self.asset_names])

        # Marginal risk contribution
        portfolio_vol = point.volatility
        marginal_contrib = (self.cov_matrix @ weights) / portfolio_vol

        # Risk contribution
        risk_contrib = weights * marginal_contrib
        risk_contrib_pct = risk_contrib / portfolio_vol

        return {name: contrib for name, contrib in zip(self.asset_names, risk_contrib_pct)}

    def get_diversification_metrics(self, point: FrontierPoint) -> Dict[str, float]:
        """
        Calculate diversification metrics.

        Args:
            point: FrontierPoint

        Returns:
            Dictionary of diversification metrics
        """
        weights = np.array([point.weights[name] for name in self.asset_names])

        # Effective number of assets (inverse Herfindahl)
        hhi = np.sum(weights ** 2)
        effective_n = 1 / hhi if hhi > 0 else 0

        # Diversification ratio
        weighted_vol = np.sum(np.sqrt(np.diag(self.cov_matrix)) * weights)
        div_ratio = weighted_vol / point.volatility if point.volatility > 0 else 0

        # Largest weight
        max_weight = np.max(np.abs(weights))

        # Weight concentration
        top_3_weight = np.sum(sorted(np.abs(weights), reverse=True)[:3])

        return {
            'Effective Number of Assets': effective_n,
            'HHI': hhi,
            'Diversification Ratio': div_ratio,
            'Max Weight': max_weight,
            'Top 3 Weight Concentration': top_3_weight
        }
