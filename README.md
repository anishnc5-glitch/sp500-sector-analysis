S&P 500 Sector Volatility & Correlation Analysis

Analyzes 5 years of daily return data across all 11 S&P 500 sector ETFs to compare risk, return, and diversification potential.

What it does
- Downloads live price data for all 11 sector ETFs (XLK, XLF, XLE, etc.) using `yfinance`
- Calculates annualized return, volatility, and Sharpe ratio for each sector
- Plots a risk/return scatter chart, correlation heatmap, and rolling 60-day volatility chart
- Exports a summary CSV

Skills applied
- Introduction to Statistics (standard deviation, correlation)
- Linear Algebra (matrix operations for covariance)
- Python: pandas, yfinance, matplotlib, seaborn

How to run
pip install yfinance pandas matplotlib seaborn
python sp500_analysis.py
