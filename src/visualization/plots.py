"""
Portfolio visualization and plotting utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class PortfolioVisualizer:
    """Visualization tools for portfolio analysis."""

    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """Initialize visualizer with style."""
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = plt.cm.Set2.colors

    def plot_price_history(self, data: pd.DataFrame,
                           title: str = "Stock Price History",
                           show_returns: bool = False) -> go.Figure:
        """
        Plot historical prices.

        Args:
            data: DataFrame with stock prices
            title: Plot title
            show_returns: Show as returns instead of prices

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        if show_returns:
            plot_data = data.pct_change().dropna() * 100
            y_title = "Daily Return (%)"
        else:
            plot_data = data
            y_title = "Price ($)"

        for col in plot_data.columns:
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data[col],
                name=col,
                mode='lines'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_title,
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def plot_correlation_heatmap(self, returns: pd.DataFrame,
                                  title: str = "Asset Correlation Matrix") -> go.Figure:
        """
        Plot correlation heatmap.

        Args:
            returns: DataFrame of returns
            title: Plot title

        Returns:
            Plotly Figure
        """
        corr_matrix = returns.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={'size': 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title=title,
            template='plotly_white'
        )

        return fig

    def plot_efficient_frontier(self, frontier_df: pd.DataFrame,
                                 optimal_point: Optional[Dict] = None,
                                 show_cal: bool = True) -> go.Figure:
        """
        Plot efficient frontier.

        Args:
            frontier_df: DataFrame with frontier points
            optimal_point: Optional dict with optimal portfolio info
            show_cal: Show Capital Allocation Line

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_df['Volatility'] * 100,
            y=frontier_df['Expected Return'] * 100,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=2)
        ))

        # Color by Sharpe ratio
        fig.add_trace(go.Scatter(
            x=frontier_df['Volatility'] * 100,
            y=frontier_df['Expected Return'] * 100,
            mode='markers',
            name='Frontier Points',
            marker=dict(
                color=frontier_df['Sharpe Ratio'],
                colorscale='Viridis',
                showscale=True,
                size=5,
                colorbar=dict(title="Sharpe Ratio")
            )
        ))

        # Optimal portfolio
        if optimal_point:
            fig.add_trace(go.Scatter(
                x=[optimal_point['volatility'] * 100],
                y=[optimal_point['return'] * 100],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(color='red', size=15, symbol='star')
            ))

        # Capital Allocation Line
        if show_cal and optimal_point:
            rf = 0.02
            max_vol = frontier_df['Volatility'].max() * 1.2
            vol_range = np.linspace(0, max_vol * 100, 50)
            cal_returns = (rf + optimal_point['sharpe'] * vol_range / 100) * 100

            fig.add_trace(go.Scatter(
                x=vol_range,
                y=cal_returns,
                mode='lines',
                name='Capital Allocation Line',
                line=dict(color='green', dash='dash')
            ))

            # Risk-free point
            fig.add_trace(go.Scatter(
                x=[0],
                y=[rf * 100],
                mode='markers',
                name='Risk-Free Rate',
                marker=dict(color='black', size=10)
            ))

        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Volatility (%)",
            yaxis_title="Expected Return (%)",
            template='plotly_white'
        )

        return fig

    def plot_portfolio_weights(self, weights: Dict[str, float],
                                title: str = "Portfolio Allocation") -> go.Figure:
        """
        Plot portfolio weights as pie chart.

        Args:
            weights: Dictionary of asset weights
            title: Plot title

        Returns:
            Plotly Figure
        """
        # Filter out zero weights
        weights = {k: v for k, v in weights.items() if v > 0.001}

        fig = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            textinfo='label+percent',
            hole=0.3,
            pull=[0.02] * len(weights)
        )])

        fig.update_layout(
            title=title,
            template='plotly_white'
        )

        return fig

    def plot_weight_comparison(self, portfolios: Dict[str, Dict[str, float]],
                                title: str = "Portfolio Comparison") -> go.Figure:
        """
        Compare portfolio weights across strategies.

        Args:
            portfolios: Dictionary of {strategy: weights_dict}
            title: Plot title

        Returns:
            Plotly Figure
        """
        # Get all assets
        all_assets = set()
        for weights in portfolios.values():
            all_assets.update(weights.keys())
        all_assets = sorted(all_assets)

        fig = go.Figure()

        for strategy, weights in portfolios.items():
            values = [weights.get(asset, 0) for asset in all_assets]
            fig.add_trace(go.Bar(
                name=strategy,
                x=all_assets,
                y=values
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Asset",
            yaxis_title="Weight",
            barmode='group',
            template='plotly_white'
        )

        return fig

    def plot_forecasts(self, historical: pd.Series,
                       forecasts: Dict[str, np.ndarray],
                       forecast_dates: pd.DatetimeIndex,
                       confidence_intervals: Optional[Dict[str, Tuple]] = None,
                       title: str = "Price Forecasts") -> go.Figure:
        """
        Plot historical data and forecasts.

        Args:
            historical: Historical price series
            forecasts: Dictionary of {model_name: forecast_array}
            forecast_dates: Dates for forecasts
            confidence_intervals: Dict of {model: (lower, upper)}
            title: Plot title

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical.values,
            name='Historical',
            line=dict(color='black', width=2)
        ))

        # Forecasts
        colors = plt.cm.tab10.colors

        for i, (model_name, forecast) in enumerate(forecasts.items()):
            color = f'rgb{colors[i % len(colors)][:3]}'

            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast,
                name=f'{model_name} Forecast',
                line=dict(color=color, dash='dash')
            ))

            # Confidence intervals
            if confidence_intervals and model_name in confidence_intervals:
                lower, upper = confidence_intervals[model_name]
                fig.add_trace(go.Scatter(
                    x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill='toself',
                    fillcolor=f'rgba({colors[i % len(colors)][:3]}, 0.2)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'{model_name} Confidence',
                    showlegend=False
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def plot_cumulative_returns(self, returns: pd.DataFrame,
                                 benchmark: Optional[pd.Series] = None,
                                 title: str = "Cumulative Returns") -> go.Figure:
        """
        Plot cumulative returns.

        Args:
            returns: DataFrame of returns
            benchmark: Optional benchmark series
            title: Plot title

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        cumulative = (1 + returns).cumprod()

        for col in cumulative.columns:
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative[col],
                name=col,
                mode='lines'
            ))

        if benchmark is not None:
            bench_cumulative = (1 + benchmark).cumprod()
            fig.add_trace(go.Scatter(
                x=bench_cumulative.index,
                y=bench_cumulative.values,
                name='Benchmark',
                mode='lines',
                line=dict(color='black', dash='dash', width=2)
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template='plotly_white'
        )

        return fig

    def plot_drawdown(self, returns: pd.Series,
                      title: str = "Portfolio Drawdown") -> go.Figure:
        """
        Plot drawdown chart.

        Args:
            returns: Portfolio returns series
            title: Plot title

        Returns:
            Plotly Figure
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            name='Drawdown',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red')
        ))

        max_dd = drawdown.min()
        fig.add_hline(y=max_dd * 100, line_dash='dash',
                     annotation_text=f'Max DD: {max_dd:.2%}')

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template='plotly_white'
        )

        return fig

    def plot_rolling_metrics(self, returns: pd.Series,
                            window: int = 252,
                            title: str = "Rolling Performance Metrics") -> go.Figure:
        """
        Plot rolling performance metrics.

        Args:
            returns: Portfolio returns
            window: Rolling window
            title: Plot title

        Returns:
            Plotly Figure
        """
        rolling_return = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_return - 0.02) / rolling_vol

        fig = make_subplots(rows=3, cols=1, subplot_titles=['Rolling Return', 'Rolling Volatility', 'Rolling Sharpe'])

        fig.add_trace(go.Scatter(x=rolling_return.index, y=rolling_return.values * 100, name='Return'), row=1, col=1)
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values * 100, name='Volatility'), row=2, col=1)
        fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name='Sharpe'), row=3, col=1)

        fig.update_yaxes(title_text='Annual Return (%)', row=1, col=1)
        fig.update_yaxes(title_text='Annual Volatility (%)', row=2, col=1)
        fig.update_yaxes(title_text='Sharpe Ratio', row=3, col=1)

        fig.update_layout(
            title=title,
            height=800,
            template='plotly_white'
        )

        return fig

    def plot_backtest_results(self, portfolio_values: pd.DataFrame,
                               benchmark_values: Optional[pd.Series] = None,
                               title: str = "Backtest Results") -> go.Figure:
        """
        Plot backtest results.

        Args:
            portfolio_values: DataFrame with portfolio values over time
            benchmark_values: Optional benchmark values
            title: Plot title

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        for col in portfolio_values.columns:
            fig.add_trace(go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values[col],
                name=col,
                mode='lines'
            ))

        if benchmark_values is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_values.index,
                y=benchmark_values.values,
                name='Benchmark',
                mode='lines',
                line=dict(color='black', dash='dash', width=2)
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def save_plot(self, fig: go.Figure, filepath: str) -> None:
        """Save plot to file."""
        fig.write_html(filepath)
        print(f"Plot saved to {filepath}")

    def plot_to_image(self, fig: go.Figure, filepath: str) -> None:
        """Save plot as static image."""
        try:
            fig.write_image(filepath)
            print(f"Image saved to {filepath}")
        except Exception as e:
            print(f"Could not save image: {e}. Install kaleido: pip install kaleido")
