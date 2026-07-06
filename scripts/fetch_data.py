"""
Script to fetch and save data for TSLA, BND, SPY.
Run: python scripts/fetch_data.py
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Configuration
TICKERS = ['TSLA', 'BND', 'SPY']
START_DATE = '2015-01-01'
END_DATE = '2026-06-30'

# Create directories
os.makedirs('../data/raw', exist_ok=True)
os.makedirs('../data/processed', exist_ok=True)

print(f"Fetching data for {TICKERS}")
print(f"Date range: {START_DATE} to {END_DATE}")

try:
    # Download data
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=True)

    if data.empty:
        raise ValueError("No data received from Yahoo Finance")

    # Extract closing prices
    prices = data['Close'][TICKERS] if len(TICKERS) > 1 else data[['Close']]
    prices.columns = TICKERS if len(TICKERS) > 1 else TICKERS

    # Save raw data
    prices.to_csv('../data/raw/stock_prices.csv')
    print(f"Saved raw prices to data/raw/stock_prices.csv")
    print(f"Shape: {prices.shape}")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")

    # Preview
    print("\nFirst 5 rows:")
    print(prices.head())
    print("\nLast 5 rows:")
    print(prices.tail())

except Exception as e:
    print(f"Error fetching data: {e}")
    raise
