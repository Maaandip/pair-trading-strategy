# pair trading strategy
# statistical arbitrage using cointegration
# stocks: HINDUNILVR and NESTLEIND (historically correlated Indian FMCG stocks)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

# ── config ───────────────────────────────────────────────────────
STOCK_A = 'HINDUNILVR.NS'
STOCK_B = 'NESTLEIND.NS'
START   = '2018-01-01'
END     = '2024-01-01'
ENTRY_Z = 1.5
EXIT_Z  = 0.25
RISK_FREE = 0.06

# ── load data ────────────────────────────────────────────────────
print("loading stock data...")
import yfinance as yf

hindunilvr_data = yf.download(STOCK_A, start=START, end=END, auto_adjust=True)
nestleind_data  = yf.download(STOCK_B, start=START, end=END, auto_adjust=True)

hindunilvr = hindunilvr_data['Close'].squeeze()
nestleind  = nestleind_data['Close'].squeeze()

df = pd.DataFrame({'HINDUNILVR': hindunilvr, 'NESTLEIND': nestleind}).dropna()
print(f"loaded {len(df)} trading days")
print(f"HINDUNILVR: {df['HINDUNILVR'].iloc[0]:.2f} → {df['HINDUNILVR'].iloc[-1]:.2f}")
print(f"NESTLEIND:  {df['NESTLEIND'].iloc[0]:.2f} → {df['NESTLEIND'].iloc[-1]:.2f}")

# ── cointegration test ───────────────────────────────────────────
print("\n=== cointegration test ===")
score, pvalue, _ = coint(df['HINDUNILVR'], df['NESTLEIND'])
print(f"p-value: {pvalue:.4f}")
if pvalue < 0.05:
    print("result: stocks are cointegrated - pair trading is valid!")
else:
    print("result: stocks are NOT cointegrated")

# ── calculate hedge ratio using OLS ─────────────────────────────
X = add_constant(df['HINDUNILVR'])
model = OLS(df['NESTLEIND'], X).fit()
hedge_ratio = model.params['HINDUNILVR']
print(f"\nhedge ratio (beta): {hedge_ratio:.4f}")

# ── calculate spread and z-score ────────────────────────────────
df['spread'] = df['NESTLEIND'] - hedge_ratio * df['HINDUNILVR']
df['spread_mean'] = df['spread'].rolling(window=30).mean()
df['spread_std']  = df['spread'].rolling(window=30).std()
df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
df = df.dropna()

# ── adf test on spread ───────────────────────────────────────────
print("\n=== ADF test on spread ===")
adf_result = adfuller(df['spread'])
print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"p-value:       {adf_result[1]:.4f}")
if adf_result[1] < 0.05:
    print("spread is stationary - mean reverting confirmed!")

# ── generate trading signals ─────────────────────────────────────
df['signal']   = 0
df['position'] = 0

position = 0
for i in range(1, len(df)):
    z = df['zscore'].iloc[i]
    if position == 0:
        if z > ENTRY_Z:
            position = -1
        elif z < -ENTRY_Z:
            position = 1
    elif position == 1:
        if z > -EXIT_Z:
            position = 0
    elif position == -1:
        if z < EXIT_Z:
            position = 0
    df.iloc[i, df.columns.get_loc('position')] = position

df['signal'] = df['position'].diff()

# ── backtest ─────────────────────────────────────────────────────
df['hindunilvr_return'] = df['HINDUNILVR'].pct_change()
df['nestleind_return']  = df['NESTLEIND'].pct_change()

df['strategy_return'] = df['position'].shift(1) * (
    df['nestleind_return'] - hedge_ratio * df['hindunilvr_return']
)
df['cumulative_return'] = (1 + df['strategy_return'].fillna(0)).cumprod()
df['cumulative_market'] = (1 + df['nestleind_return'].fillna(0)).cumprod()

total_return = df['cumulative_return'].iloc[-1] - 1
num_trades   = (df['signal'] != 0).sum()
sharpe       = (df['strategy_return'].mean() * 252) / \
               (df['strategy_return'].std() * np.sqrt(252))
max_dd       = (df['cumulative_return'] /
                df['cumulative_return'].cummax() - 1).min()

print("\n=== backtest results ===")
print(f"total return:  {total_return*100:.2f}%")
print(f"sharpe ratio:  {sharpe:.4f}")
print(f"max drawdown:  {max_dd*100:.2f}%")
print(f"total trades:  {num_trades}")

# ── plots ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(13, 16))

ax  = axes[0]
ax2 = ax.twinx()
ax.plot(df.index,  df['HINDUNILVR'], color='steelblue', label='HINDUNILVR', linewidth=1)
ax2.plot(df.index, df['NESTLEIND'],  color='orange',    label='NESTLEIND',  linewidth=1)
ax.set_ylabel('HINDUNILVR price', color='steelblue')
ax2.set_ylabel('NESTLEIND price', color='orange')
ax.set_title('HINDUNILVR vs NESTLEIND stock prices', fontsize=12)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

axes[1].plot(df.index, df['spread'], color='purple', linewidth=1, label='spread')
axes[1].plot(df.index, df['spread_mean'], color='black',
             linewidth=1, linestyle='--', label='30d mean')
axes[1].set_title('spread (NESTLEIND - β × HINDUNILVR)', fontsize=12)
axes[1].legend()

axes[2].plot(df.index, df['zscore'], color='darkgreen', linewidth=1, label='z-score')
axes[2].axhline( ENTRY_Z, color='red',  linestyle='--', alpha=0.7, label=f'entry +{ENTRY_Z}')
axes[2].axhline(-ENTRY_Z, color='red',  linestyle='--', alpha=0.7, label=f'entry -{ENTRY_Z}')
axes[2].axhline( EXIT_Z,  color='blue', linestyle=':',  alpha=0.7, label=f'exit +{EXIT_Z}')
axes[2].axhline(-EXIT_Z,  color='blue', linestyle=':',  alpha=0.7, label=f'exit -{EXIT_Z}')
axes[2].axhline(0, color='black', linewidth=0.8)

long_entry  = df[df['signal'] ==  2]
short_entry = df[df['signal'] == -2]
exits       = df[(df['signal'] ==  1) | (df['signal'] == -1)]

axes[2].scatter(long_entry.index,  long_entry['zscore'],
                color='green', marker='^', s=80, zorder=5, label='long entry')
axes[2].scatter(short_entry.index, short_entry['zscore'],
                color='red',   marker='v', s=80, zorder=5, label='short entry')
axes[2].scatter(exits.index, exits['zscore'],
                color='black', marker='x', s=60, zorder=5, label='exit')
axes[2].set_title('z-score with entry/exit signals', fontsize=12)
axes[2].legend(fontsize=8, ncol=3)

axes[3].plot(df.index, df['cumulative_return'],
             color='green', linewidth=1.5, label='pair trading strategy')
axes[3].plot(df.index, df['cumulative_market'],
             color='grey', linewidth=1, linestyle='--', label='buy & hold NESTLEIND')
axes[3].axhline(1, color='black', linewidth=0.8)
axes[3].set_title(
    f'equity curve  |  return: {total_return*100:.1f}%  '
    f'sharpe: {sharpe:.2f}  max dd: {max_dd*100:.1f}%',
    fontsize=12
)
axes[3].legend()

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

plt.suptitle('pair trading strategy — HINDUNILVR / NESTLEIND',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/pair_trading.png', dpi=150)
plt.close()
print("\nsaved output/pair_trading.png")
print("\ndone!")