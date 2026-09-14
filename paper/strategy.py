"""
Estrategias de momentum sobre ETFs. Tres variantes, una sola máquina.

Regla común: el primer día hábil de cada mes, con cierres hasta el día anterior,
ranking por retorno de los últimos `lookback` días; comprar los `top_n` mejores
a partes iguales si superan un filtro; el resto en efectivo.

  MOM6  — lookback 6m, filtro: retorno > 0.                      (base)
  VOLT  — como MOM6, pero la exposición total se escala para que la cartera
          apunte a `vol_target` de volatilidad anual (Moreira & Muir, 2017).
  DUAL  — lookback 12m, filtro: retorno > retorno de T-bills (BIL)
          (dual momentum, Antonacci 2014).

Backtest y paper trading llaman a la misma `target_weights`, así lo que se
probó es exactamente lo que se opera.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC"]
CASH_PROXY = "BIL"          # ETF de T-bills a 1-3 meses: "el retorno del efectivo"
COST_BPS = 10
TRADING_DAYS = 252


@dataclass(frozen=True)
class Strategy:
    name: str
    lookback: int            # días hábiles
    top_n: int = 3
    filter: str = "zero"     # "zero": retorno > 0 | "tbill": retorno > retorno de BIL
    vol_target: float | None = None   # None = sin control; 0.10 = apuntar a 10% anual
    vol_window: int = 63     # ventana para medir la volatilidad realizada


STRATEGIES = {
    "MOM6": Strategy("MOM6", lookback=126),
    "VOLT": Strategy("VOLT", lookback=126, vol_target=0.10),
    "DUAL": Strategy("DUAL", lookback=252, filter="tbill"),
}


def symbols_needed(s: Strategy) -> list[str]:
    return UNIVERSE + ([CASH_PROXY] if s.filter == "tbill" else [])


# ---------------------------------------------------------------------------
# Señal
# ---------------------------------------------------------------------------

def momentum_scores(prices: pd.DataFrame, lookback: int) -> pd.Series:
    if len(prices) < lookback + 1:
        raise ValueError(f"hacen falta al menos {lookback + 1} días de precios")
    return prices.iloc[-1] / prices.iloc[-1 - lookback] - 1


def target_weights(prices: pd.DataFrame, s: Strategy) -> pd.Series:
    """
    Pesos objetivo sobre UNIVERSE (suman <= 1; el resto es efectivo).
    prices: cierres ajustados hasta el ÚLTIMO cierre disponible antes de operar,
    con las columnas de symbols_needed(s).
    """
    scores = momentum_scores(prices, s.lookback)
    hurdle = scores[CASH_PROXY] if s.filter == "tbill" else 0.0
    ranked = scores[UNIVERSE].dropna().sort_values(ascending=False)
    winners = ranked.head(s.top_n)
    winners = winners[winners > hurdle]

    w = pd.Series(0.0, index=UNIVERSE)
    if len(winners) == 0:
        return w
    w[winners.index] = 1.0 / s.top_n

    if s.vol_target is not None:
        basket = prices[winners.index].pct_change().iloc[-s.vol_window:].mean(axis=1)
        realized = basket.std() * np.sqrt(TRADING_DAYS)
        exposure = min(1.0, s.vol_target / realized) if realized > 0 else 1.0
        w = w * exposure
    return w


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.Series(1, index=index).groupby([index.year, index.month]).head(1).index


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest(prices: pd.DataFrame, s: Strategy, cost_bps: float = COST_BPS) -> dict:
    """Señal con cierres hasta ayer, ejecución al cierre del día de rebalanceo, costes por rotación."""
    rets = prices[UNIVERSE].pct_change().fillna(0.0)
    dates = prices.index
    rb = set(rebalance_dates(dates))
    warmup = max(s.lookback, s.vol_window) + 1
    w = pd.Series(0.0, index=UNIVERSE)
    equity, hist, eq = [], {}, 1.0
    for i, d in enumerate(dates):
        if i > warmup and d in rb:
            new_w = target_weights(prices.iloc[:i], s)
            eq *= 1 - (new_w - w).abs().sum() * cost_bps / 1e4
            w = new_w
            hist[d] = w.copy()
        eq *= 1 + (w * rets.iloc[i]).sum()
        equity.append(eq)
    equity = pd.Series(equity, index=dates, name=s.name)
    bench = (1 + rets["SPY"]).cumprod().rename("SPY")
    return {"equity": equity, "benchmark": bench, "weights": pd.DataFrame(hist).T}


def metrics(equity: pd.Series) -> dict:
    r = equity.pct_change().dropna()
    n_years = len(r) / TRADING_DAYS
    return {
        "CAGR": equity.iloc[-1] ** (1 / n_years) - 1,
        "vol": r.std() * np.sqrt(TRADING_DAYS),
        "sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS),
        "max_drawdown": (equity / equity.cummax() - 1).min(),
    }
