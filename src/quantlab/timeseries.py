"""
Series de tiempo: autocorrelación y GARCH(1,1) construido a mano.

Idea central del módulo: la DIRECCIÓN de los retornos es casi impredecible,
pero su VOLATILIDAD sí tiene memoria (clustering). GARCH modela esa memoria.

GARCH(1,1):  sigma²_t = omega + alpha * r²_{t-1} + beta * sigma²_{t-1}
  - omega: nivel base de varianza.
  - alpha: cuánto reacciona la volatilidad al shock de ayer.
  - beta:  cuánta volatilidad de ayer "persiste" hoy.
  - alpha + beta < 1 para que la volatilidad vuelva a su media a largo plazo.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Autocorrelación
# ---------------------------------------------------------------------------

def acf(x: pd.Series | np.ndarray, max_lag: int = 30) -> np.ndarray:
    """
    Autocorrelación para lags 1..max_lag: correlación de la serie con ella misma
    desplazada k días. ~0 => el pasado no informa el presente (para esa métrica).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = (x**2).sum()
    return np.array([(x[k:] * x[:-k]).sum() / denom for k in range(1, max_lag + 1)])


def acf_confidence_band(n: int, confidence: float = 0.95) -> float:
    """Banda ±z/sqrt(n): fuera de ella la autocorrelación es 'significativa'."""
    from scipy.stats import norm
    return norm.ppf(1 - (1 - confidence) / 2) / np.sqrt(n)


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------

def _garch_variance(r: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Recorre la serie y calcula la varianza condicional día a día."""
    n = len(r)
    var = np.empty(n)
    var[0] = r.var()  # arranque: varianza incondicional muestral
    for t in range(1, n):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
    return var


def _garch_neg_loglik(params: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    var = _garch_variance(r, omega, alpha, beta)
    var = np.maximum(var, 1e-12)
    return 0.5 * np.sum(np.log(2 * np.pi * var) + r**2 / var)


def garch_fit(returns: pd.Series) -> dict:
    """
    Ajusta GARCH(1,1) por máxima verosimilitud: busca (omega, alpha, beta) que
    hacen más probable haber observado exactamente estos retornos.
    Devuelve dict con parámetros, varianza condicional y vol anualizada.
    """
    r = returns.to_numpy(dtype=float)
    r = r - r.mean()
    uncond = r.var()
    x0 = np.array([0.05 * uncond, 0.08, 0.90])
    res = minimize(_garch_neg_loglik, x0, args=(r,), method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10})
    omega, alpha, beta = res.x
    var = _garch_variance(r, omega, alpha, beta)
    cond_vol = pd.Series(np.sqrt(var) * np.sqrt(TRADING_DAYS), index=returns.index, name="garch_vol")
    return {
        "omega": float(omega), "alpha": float(alpha), "beta": float(beta),
        "persistence": float(alpha + beta),
        "long_run_vol": float(np.sqrt(omega / (1 - alpha - beta) * TRADING_DAYS)),
        "cond_vol": cond_vol,
        "last_var": float(var[-1]),
        "last_ret2": float(r[-1] ** 2),
    }


def garch_forecast(fit: dict, horizon: int = 30) -> np.ndarray:
    """
    Pronóstico de volatilidad anualizada para los próximos `horizon` días.
    Converge hacia long_run_vol: la volatilidad extrema no dura para siempre.
    """
    omega, alpha, beta = fit["omega"], fit["alpha"], fit["beta"]
    var = omega + alpha * fit["last_ret2"] + beta * fit["last_var"]
    out = [var]
    for _ in range(horizon - 1):
        var = omega + (alpha + beta) * var
        out.append(var)
    return np.sqrt(np.array(out) * TRADING_DAYS)


def simulate_garch(
    omega: float, alpha: float, beta: float,
    n_days: int = TRADING_DAYS, n_paths: int = 1, seed: int | None = 42,
    s0: float | None = None, mu_daily: float = 0.0,
) -> np.ndarray:
    """
    Genera retornos con volatilidad GARCH. Los shocks son normales, pero como
    la varianza cambia día a día, los retornos resultantes tienen colas gordas
    y clustering: dos cosas que el GBM no puede producir.
    Devuelve retornos (n_days, n_paths), o precios (n_days+1, n_paths) si s0.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_days, n_paths))
    var = np.full(n_paths, omega / (1 - alpha - beta))
    rets = np.empty((n_days, n_paths))
    for t in range(n_days):
        rets[t] = mu_daily + np.sqrt(var) * z[t]
        var = omega + alpha * (rets[t] - mu_daily) ** 2 + beta * var
    if s0 is None:
        return rets
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(rets, axis=0)])
    return s0 * np.exp(log_paths)


# ---------------------------------------------------------------------------
# Régimen de volatilidad (módulo de alertas del capstone)
# ---------------------------------------------------------------------------

def volatility_regime(fit: dict, low_pct: float = 33, high_pct: float = 67) -> dict:
    """
    Clasifica el momento actual comparando la vol GARCH de hoy con su propia
    historia: 'bajo' / 'normal' / 'alto'. Base de la alerta del capstone.
    """
    cv = fit["cond_vol"]
    today = float(cv.iloc[-1])
    lo, hi = np.percentile(cv, [low_pct, high_pct])
    regime = "bajo" if today < lo else "alto" if today > hi else "normal"
    percentile = float((cv < today).mean() * 100)
    return {"vol_today": today, "regime": regime, "percentile": percentile,
            "threshold_low": float(lo), "threshold_high": float(hi)}
