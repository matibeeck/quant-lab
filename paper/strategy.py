"""
Estrategia: momentum mensual sobre ETFs.

Regla (fijada antes de mirar resultados):
- Universo: 8 ETFs con perfiles distintos.
- El primer día hábil de cada mes, con precios de cierre del mes anterior,
  ranking por retorno de los últimos LOOKBACK meses (~126 días hábiles).
- Comprar los TOP_N mejores a partes iguales, SOLO si su retorno es positivo.
  Lo que no se invierte se queda en efectivo.
- Mantener hasta el siguiente rebalanceo.

Todo lo que decide la estrategia está en `target_weights`. Backtest y paper
trading llaman a la misma función, así no puede haber diferencias entre lo
que se probó y lo que se opera.
"""

import numpy as np
import pandas as pd

UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC"]
LOOKBACK_DAYS = 126   # ~6 meses hábiles
TOP_N = 3
COST_BPS = 10         # coste por operación en el backtest: 0.10% del importe


def momentum_scores(prices: pd.DataFrame, lookback: int = LOOKBACK_DAYS) -> pd.Series:
    """Retorno de los últimos `lookback` días hábiles, con el último precio disponible."""
    if len(prices) < lookback + 1:
        raise ValueError(f"hacen falta al menos {lookback + 1} días de precios")
    return prices.iloc[-1] / prices.iloc[-1 - lookback] - 1


def target_weights(prices: pd.DataFrame, top_n: int = TOP_N, lookback: int = LOOKBACK_DAYS) -> pd.Series:
    """
    Pesos objetivo (suman <= 1; el resto es efectivo).
    prices: DataFrame de cierres ajustados, columnas = tickers, hasta el ÚLTIMO
    cierre disponible ANTES de operar (nunca el cierre del día en que se opera).
    """
    scores = momentum_scores(prices, lookback).dropna()
    winners = scores.sort_values(ascending=False).head(top_n)
    winners = winners[winners > 0]
    w = pd.Series(0.0, index=prices.columns)
    if len(winners):
        w[winners.index] = 1.0 / top_n
    return w


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Primer día hábil de cada mes dentro del índice."""
    s = pd.Series(index=index, data=1)
    return s.groupby([index.year, index.month]).head(1).index


def backtest(prices: pd.DataFrame, top_n: int = TOP_N, lookback: int = LOOKBACK_DAYS,
             cost_bps: float = COST_BPS) -> dict:
    """
    Backtest honesto:
    - Señal con cierres hasta el día ANTERIOR al rebalanceo (sin look-ahead).
    - Ejecución al cierre del día de rebalanceo.
    - Coste proporcional al importe rotado.
    Devuelve equity de la estrategia y de SPY, y el historial de pesos.
    """
    rets = prices.pct_change().fillna(0.0)
    dates = prices.index
    rb = set(rebalance_dates(dates))
    w = pd.Series(0.0, index=prices.columns)
    equity, weights_hist = [], {}
    eq = 1.0
    for i, d in enumerate(dates):
        if i > lookback and d in rb:
            new_w = target_weights(prices.iloc[:i], top_n, lookback)   # iloc[:i] = hasta AYER
            turnover = (new_w - w).abs().sum()
            eq *= 1 - turnover * cost_bps / 1e4
            w = new_w
            weights_hist[d] = w.copy()
        eq *= 1 + (w * rets.iloc[i]).sum()
        equity.append(eq)
    equity = pd.Series(equity, index=dates, name="strategy")
    bench = (1 + rets["SPY"]).cumprod().rename("SPY")
    return {"equity": equity, "benchmark": bench, "weights": pd.DataFrame(weights_hist).T}


def metrics(equity: pd.Series) -> dict:
    r = equity.pct_change().dropna()
    n_years = len(r) / 252
    cagr = equity.iloc[-1] ** (1 / n_years) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252)
    dd = (equity / equity.cummax() - 1).min()
    return {"CAGR": cagr, "vol": vol, "sharpe": sharpe, "max_drawdown": dd}
