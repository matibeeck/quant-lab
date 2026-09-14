"""Backtest rápido de la misma regla con datos de Yahoo, para saber qué esperar del paper trading."""
import sys
import pandas as pd
sys.path.insert(0, ".")
from strategy import UNIVERSE, backtest, metrics

import yfinance as yf
raw = yf.download(UNIVERSE, start="2007-01-01", auto_adjust=False, progress=False)
prices = raw["Adj Close"].dropna()

res = backtest(prices)
tab = pd.DataFrame({"Momentum ETF": metrics(res["equity"]), "SPY buy&hold": metrics(res["benchmark"])}).T
print(f"Período: {prices.index[0].date()} – {prices.index[-1].date()}\n")
print(tab.to_string(float_format=lambda x: f"{x:.2%}" if abs(x) < 5 else f"{x:.2f}"))
print("\nÚltimos pesos:\n", res["weights"].tail(3).round(2).to_string())

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
pd.concat([res["equity"], res["benchmark"]], axis=1).plot(logy=True, figsize=(11, 5), title="Momentum ETF vs SPY (escala log)")
plt.savefig("backtest.png", dpi=150, bbox_inches="tight"); print("\ngráfico: paper/backtest.png")
