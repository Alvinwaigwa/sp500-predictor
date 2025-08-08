Professional RSI Strategy Backtest
Overview
This project implements a professional backtest of a trading strategy based on the Relative Strength Index (RSI) applied to the S&P 500 Index from 2020 to 2023.
The strategy buys when the RSI indicates the market is oversold (below 30) and sells when overbought (above 70). It evaluates performance metrics such as Sharpe ratio, Sortino ratio, max drawdown, total returns, win rate, and number of trades, and visualizes the equity curve alongside buy/sell signals.

Features
Download daily S&P 500 price data from Yahoo Finance.
Calculate RSI using Wilder's EMA method.
Generate buy/sell signals based on RSI thresholds.
Backtest the strategy with transaction costs included.
Calculate key performance metrics.
Visualize equity curve and trading signals on price chart.
Clear and modular Python code with comments for easy customization.

Requirements
Python 3.7+
pandas
numpy
matplotlib
yfinance