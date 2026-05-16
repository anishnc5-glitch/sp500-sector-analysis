# sp500_analysis.py
# I wanted to see which sectors of the stock market are the most volatile
# and how correlated they are with each other. Used stuff from stats class
# (std deviation, correlation) and linear algebra (covariance matrix).

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# these are the 11 sector ETFs that make up the S&P 500
sectors = {
    "XLK":  "Tech",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Healthcare",
    "XLI":  "Industrials",
    "XLP":  "Consumer Staples",
    "XLY":  "Consumer Discret.",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLB":  "Materials",
    "XLC":  "Communication",
}

# ------------------------------------------------------------------
# pull 3 years of price data
# ------------------------------------------------------------------

print("grabbing price data...")
tickers = list(sectors.keys())
prices = yf.download(tickers, period="3y", auto_adjust=True, progress=False)["Close"]
prices.columns = [sectors[t] for t in prices.columns]

# daily percent change -- this is what we actually care about
daily_returns = prices.pct_change().dropna()

# ------------------------------------------------------------------
# basic stats -- annualized return and standard deviation
# std dev is basically how "jumpy" each sector is day to day
# annualize by multiplying by sqrt(252) -- there are ~252 trading days/year
# ------------------------------------------------------------------

avg_return = daily_returns.mean() * 252        # annualized
volatility = daily_returns.std() * (252**0.5)  # annualized std dev

summary = pd.DataFrame({
    "avg annual return": (avg_return * 100).round(2),
    "volatility (std dev)": (volatility * 100).round(2),
}).sort_values("volatility (std dev)", ascending=False)

print("\n--- sector stats ---")
print(summary.to_string())

# ------------------------------------------------------------------
# covariance matrix using linear algebra
# cov(X, Y) = E[(X - mean_X)(Y - mean_Y)]
# in matrix form: (1/n) * X^T * X  (where X is mean-centered)
# ------------------------------------------------------------------

n = len(daily_returns)
mean_centered = daily_returns - daily_returns.mean()

# manual covariance matrix -- same as daily_returns.cov() but doing it by hand
# to show the matrix math from linear algebra class
cov_matrix = (mean_centered.T @ mean_centered) / (n - 1)
cov_df = pd.DataFrame(cov_matrix, index=daily_returns.columns, columns=daily_returns.columns)

print("\ncovariance matrix (annualized):")
print((cov_df * 252).round(6).to_string())

# ------------------------------------------------------------------
# chart 1: bar chart of volatility by sector
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#e74c3c" if v > volatility.mean() else "#3498db" for v in volatility]
volatility.sort_values().plot(kind="barh", color=colors[::-1], ax=ax)

ax.axvline(volatility.mean(), color="black", linestyle="--", linewidth=1, label="average")
ax.set_xlabel("annualized std deviation")
ax.set_title("which sectors are the most volatile? (3yr)")
ax.legend()
plt.tight_layout()
plt.savefig("sector_volatility.png", dpi=150)
plt.close()
print("\nsaved: sector_volatility.png")

# ------------------------------------------------------------------
# chart 2: correlation heatmap
# correlation = covariance(X,Y) / (std(X) * std(Y))
# goes from -1 (opposite) to +1 (move together)
# ------------------------------------------------------------------

corr_matrix = daily_returns.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    ax=ax
)
ax.set_title("sector return correlations\n(+1 = move together, -1 = move opposite)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()
print("saved: correlation_heatmap.png")

# ------------------------------------------------------------------
# chart 3: risk vs return scatter
# sectors in the top-left corner are best (high return, low risk)
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))
for sector in avg_return.index:
    ax.scatter(volatility[sector] * 100, avg_return[sector] * 100, s=80, zorder=3)
    ax.annotate(sector, (volatility[sector] * 100, avg_return[sector] * 100),
                textcoords="offset points", xytext=(5, 3), fontsize=9)

ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("volatility / risk (std dev %)")
ax.set_ylabel("avg annual return (%)")
ax.set_title("risk vs return by sector -- ideally want top-left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("risk_vs_return.png", dpi=150)
plt.close()
print("saved: risk_vs_return.png")

print("\ndone! check the 3 png files")
