"""
Streamlit dashboard for portfolio management.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import plotly.graph_objects as go


def create_dashboard():
    """Create the Streamlit dashboard."""
    st.set_page_config(
        page_title="Portfolio Optimization Dashboard",
        page_icon="📈",
        layout="wide"
    )

    st.title("Time Series Forecasting for Portfolio Management Optimization")

    # Sidebar
    st.sidebar.header("Configuration")

    # Configuration inputs
    st.sidebar.subheader("Stock Selection")
    default_tickers = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,JPM,V,UNH"
    ticker_input = st.sidebar.text_area(
        "Enter tickers (comma-separated)",
        value=default_tickers
    )

    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=730))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())

    st.sidebar.subheader("Forecast Settings")
    forecast_days = st.sidebar.slider("Forecast Days", 7, 180, 30)

    st.sidebar.subheader("Optimization Settings")
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.0) / 100

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview", "Forecasting", "Portfolio Optimization",
        "Risk Analysis", "Portfolio Analysis"
    ])

    with tab1:
        st.header("Data Overview")

        if st.button("Fetch Data"):
            with st.spinner("Fetching stock data..."):
                try:
                    # Import here to avoid issues when dependencies not installed
                    from ..data.stock_data import StockDataFetcher

                    tickers = [t.strip().upper() for t in ticker_input.split(",")]
                    fetcher = StockDataFetcher(
                        tickers=tickers,
                        start_date=str(start_date),
                        end_date=str(end_date)
                    )
                    data = fetcher.fetch_data()

                    if data is not None and not data.empty:
                        st.session_state['data'] = data
                        st.session_state['fetcher'] = fetcher
                        st.session_state['tickers'] = tickers
                        st.success(f"Successfully fetched data for {len(tickers)} stocks")
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

        if 'data' in st.session_state:
            data = st.session_state['data']
            fetcher = st.session_state.get('fetcher')

            # Price chart
            st.subheader("Historical Prices")
            fig = go.Figure()
            for col in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col, mode='lines'))
            fig.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Returns
            if fetcher:
                try:
                    returns = fetcher.calculate_returns()
                    st.session_state['returns'] = returns

                    # Correlation matrix
                    st.subheader("Correlation Matrix")
                    corr = returns.corr()
                    fig = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        colorscale='RdBu_r',
                        zmid=0
                    ))
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

                    # Summary statistics
                    st.subheader("Summary Statistics")
                    stats = fetcher.get_summary_stats()
                    st.dataframe(stats.style.format({
                        'Mean Daily Return': '{:.4%}',
                        'Daily Volatility': '{:.4%}',
                        'Annualized Return': '{:.2%}',
                        'Annualized Volatility': '{:.2%}',
                        'Sharpe Ratio': '{:.3f}',
                        'Max Return': '{:.4%}',
                        'Min Return': '{:.4%}',
                        'Skewness': '{:.3f}',
                        'Kurtosis': '{:.3f}'
                    }))
                except Exception as e:
                    st.warning(f"Could not calculate returns: {e}")

    with tab2:
        st.header("Time Series Forecasting")

        if 'data' not in st.session_state:
            st.warning("Please fetch data first in the Data Overview tab.")
        else:
            data = st.session_state['data']
            ticker_to_forecast = st.selectbox("Select ticker to forecast", data.columns.tolist())

            model_type = st.multiselect(
                "Select forecasting models",
                ["ARIMA", "Prophet", "LSTM"],
                default=["ARIMA"]
            )

            if st.button("Generate Forecasts"):
                if ticker_to_forecast in data.columns:
                    with st.spinner("Generating forecasts..."):
                        series = data[ticker_to_forecast].dropna()
                        forecasts = {}
                        confidence_intervals = {}

                        try:
                            # ARIMA
                            if "ARIMA" in model_type:
                                try:
                                    from ..models.arima_model import ARIMAForecaster
                                    arima = ARIMAForecaster(order=(1, 1, 1))
                                    arima.fit(series)
                                    forecast, conf_int = arima.predict_with_confidence(steps=forecast_days)
                                    forecasts['ARIMA'] = forecast.values
                                    confidence_intervals['ARIMA'] = (
                                        conf_int.iloc[:, 0].values,
                                        conf_int.iloc[:, 1].values
                                    )
                                except Exception as e:
                                    st.warning(f"ARIMA failed: {e}")

                            # Prophet
                            if "Prophet" in model_type:
                                try:
                                    from ..models.prophet_model import ProphetForecaster
                                    prophet = ProphetForecaster()
                                    prophet.fit(series)
                                    forecast_df = prophet.predict(periods=forecast_days)
                                    forecasts['Prophet'] = forecast_df['yhat'].values
                                except Exception as e:
                                    st.warning(f"Prophet failed: {e}")

                            # Store forecasts
                            if forecasts:
                                st.session_state['forecasts'] = forecasts
                                st.session_state['confidence_intervals'] = confidence_intervals
                                st.session_state['forecast_ticker'] = ticker_to_forecast
                                st.success(f"Generated forecasts for {ticker_to_forecast}")

                        except Exception as e:
                            st.error(f"Forecasting error: {e}")

            # Display forecasts
            if 'forecasts' in st.session_state:
                forecasts = st.session_state.get('forecasts', {})
                confidence_intervals = st.session_state.get('confidence_intervals', {})
                ticker = st.session_state.get('forecast_ticker')
                series = data[ticker].dropna()

                st.subheader(f"Forecasts for {ticker}")

                fig = go.Figure()

                # Historical
                fig.add_trace(go.Scatter(
                    x=series.index[-60:],
                    y=series.values[-60:],
                    name="Historical",
                    line=dict(color='black', width=2)
                ))

                # Forecasts
                last_date = series.index[-1]
                forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)

                for model, forecast in forecasts.items():
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast,
                        name=f"{model} Forecast",
                        line=dict(dash='dash')
                    ))

                fig.update_layout(
                    template='plotly_white',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Portfolio Optimization")

        if 'returns' not in st.session_state:
            st.warning("Please fetch data first in the Data Overview tab.")
        else:
            returns = st.session_state['returns']

            st.subheader("Optimization Settings")
            optimization_method = st.selectbox(
                "Optimization Method",
                ["Maximum Sharpe Ratio", "Minimum Volatility", "Equal Weight", "Risk Parity"]
            )

            max_weight = st.slider("Maximum Weight per Asset (%)", 0, 100, 100) / 100

            if st.button("Optimize Portfolio"):
                with st.spinner("Optimizing portfolio..."):
                    try:
                        from ..optimization.mean_variance import MeanVarianceOptimizer

                        optimizer = MeanVarianceOptimizer(returns, risk_free_rate=risk_free_rate)

                        if optimization_method == "Maximum Sharpe Ratio":
                            result = optimizer.maximize_sharpe_ratio(max_weight=max_weight if max_weight < 1 else None)
                        elif optimization_method == "Minimum Volatility":
                            result = optimizer.minimize_volatility(max_weight=max_weight if max_weight < 1 else None)
                        elif optimization_method == "Equal Weight":
                            result = optimizer.equal_weight()
                        elif optimization_method == "Risk Parity":
                            result = optimizer.risk_parity()

                        st.session_state['optimization_result'] = result
                        st.session_state['optimizer'] = optimizer

                        st.success(f"Optimization complete using {optimization_method}")

                    except Exception as e:
                        st.error(f"Optimization error: {e}")

            # Display results
            if 'optimization_result' in st.session_state:
                result = st.session_state['optimization_result']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Return", f"{result.expected_return:.2%}")
                with col2:
                    st.metric("Expected Volatility", f"{result.expected_volatility:.2%}")
                with col3:
                    st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.3f}")

                # Weights chart
                st.subheader("Portfolio Weights")
                weights = result.weights
                weights_dict = {k: v for k, v in weights.items() if v > 0.001}

                fig = go.Figure(data=[go.Pie(
                    labels=list(weights_dict.keys()),
                    values=list(weights_dict.values()),
                    textinfo='label+percent',
                    hole=0.3
                )])
                st.plotly_chart(fig, use_container_width=True)

                # Weights table
                weight_df = pd.DataFrame({
                    'Asset': list(weights.keys()),
                    'Weight': list(weights.values())
                }).sort_values('Weight', ascending=False)
                st.dataframe(weight_df.style.format({'Weight': '{:.4f}'}))

    with tab4:
        st.header("Risk Analysis")

        if 'returns' not in st.session_state:
            st.warning("Please fetch data first in the Data Overview tab.")
        else:
            # Risk metrics calculator
            if st.button("Calculate Risk Metrics"):
                with st.spinner("Calculating risk metrics..."):
                    try:
                        from ..optimization.risk_metrics import RiskMetrics

                        # Equal weight portfolio
                        returns = st.session_state['returns']
                        portfolio_returns = returns.mean(axis=1)

                        risk_metrics = RiskMetrics(portfolio_returns)
                        metrics_df = risk_metrics.get_all_metrics()

                        st.session_state['risk_metrics'] = metrics_df

                        # Drawdown
                        fig = go.Figure()
                        cumulative = (1 + portfolio_returns).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdown = (cumulative - running_max) / running_max

                        fig.add_trace(go.Scatter(
                            x=drawdown.index,
                            y=drawdown.values * 100,
                            name='Drawdown',
                            fill='tozeroy',
                            fillcolor='rgba(255, 0, 0, 0.3)',
                            line=dict(color='red')
                        ))

                        fig.update_layout(
                            title="Portfolio Drawdown",
                            xaxis_title="Date",
                            yaxis_title="Drawdown (%)",
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error calculating risk metrics: {e}")

            if 'risk_metrics' in st.session_state:
                st.subheader("Risk Metrics Summary")
                st.dataframe(st.session_state['risk_metrics'])

    with tab5:
        st.header("Portfolio Analysis Report")

        if 'optimization_result' not in st.session_state:
            st.warning("Please optimize a portfolio first in the Portfolio Optimization tab.")
        else:
            result = st.session_state['optimization_result']
            optimizer = st.session_state['optimizer']

            st.subheader("Portfolio Summary")

            # Download report
            if st.button("Generate Report"):
                report = f"""
                ## Portfolio Optimization Report
                **Method:** {result.method}
                **Date Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                ### Performance Metrics
                - Expected Annual Return: {result.expected_return:.2%}
                - Expected Annual Volatility: {result.expected_volatility:.2%}
                - Sharpe Ratio: {result.sharpe_ratio:.3f}

                ### Portfolio Weights
                """
                for asset, weight in sorted(result.weights.items(), key=lambda x: -x[1]):
                    if weight > 0.001:
                        report += f"\n- {asset}: {weight:.2%}"

                st.markdown(report)

                st.download_button(
                    "Download Report",
                    report,
                    file_name="portfolio_report.md"
                )


if __name__ == "__main__":
    create_dashboard()
