"""Backtest de las tres variantes con datos de Yahoo, lado a lado."""
import sys
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from strategy import STRATEGIES, UNIVERSE, CASH_PROXY, backtest, metrics

import yfinance as yf
raw = yf.download(UNIVERSE + [CASH_PROXY], start="2007-01-01", auto_adjust=False, progress=False)
prices = raw["Adj Close"].dropna()          # BIL empieza en mayo 2007

rows, curves = {}, {}
for name, s in STRATEGIES.items():
    res = backtest(prices, s)
    rows[name] = metrics(res["equity"]); curves[name] = res["equity"]
rows["SPY buy&hold"] = metrics(res["benchmark"]); curves["SPY"] = res["benchmark"]

tab = pd.DataFrame(rows).T
print(f"Período: {prices.index[0].date()} – {prices.index[-1].date()}\n")
print(tab.to_string(formatters={"CAGR": "{:.2%}".format, "vol": "{:.2%}".format,
                                "sharpe": "{:.2f}".format, "max_drawdown": "{:.2%}".format}))

pd.DataFrame(curves).plot(logy=True, figsize=(11, 5), title="Variantes de momentum vs SPY (escala log)")
plt.savefig("backtest.png", dpi=150, bbox_inches="tight"); print("\ngráfico: paper/backtest.png")
