"""Setup script for the package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="portfolio-optimization",
    version="1.0.0",
    author="Portfolio Optimization Team",
    author_email="team@portfolio-opt.com",
    description="Time Series Forecasting for Portfolio Management Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abigiyacodehub/Time-Series-Forecasting-for-Portfolio-Management-Optimization.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "yfinance>=0.2.28",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.18.0",
        "cvxpy>=1.4.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "full": [
            "prophet>=1.1.4",
            "tensorflow>=2.14.0",
            "streamlit>=1.28.0",
            "pypfopt>=1.5.5",
            "ta>=0.11.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "portfolio-opt=main:main",
        ],
    },
)
