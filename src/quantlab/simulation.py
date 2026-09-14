"""
Simulación de precios: Monte Carlo.

La idea de Monte Carlo: no sabemos el futuro, pero podemos generar miles de
futuros posibles que sean consistentes con lo que hemos observado, y mirar la
distribución de resultados en lugar de una sola predicción.

Convenciones:
- mu y sigma son ANUALES (retorno esperado y volatilidad de los log-retornos).
- Los resultados son arrays numpy de forma (n_days + 1, n_paths): fila 0 es hoy.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Parámetros a partir del histórico
# ---------------------------------------------------------------------------

def estimate_params(log_rets: pd.Series, periods: int = TRADING_DAYS) -> tuple[float, float]:
    """
    Devuelve (mu, sigma) anualizados a partir de log-retornos diarios.
    mu = media diaria * 252;  sigma = std diaria * sqrt(252).
    """
    mu = log_rets.mean() * periods
    sigma = log_rets.std() * np.sqrt(periods)
    return float(mu), float(sigma)


# ---------------------------------------------------------------------------
# Movimiento browniano geométrico (GBM)
# ---------------------------------------------------------------------------

def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    n_days: int = TRADING_DAYS,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> np.ndarray:
    """
    S_{t+1} = S_t * exp( (mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z ),  Z ~ N(0, 1)

    - (mu - sigma^2/2) * dt: la parte "esperada" del movimiento diario.
      El -sigma^2/2 corrige que la media de exp(X) no es exp(media de X).
    - sigma * sqrt(dt) * Z: la parte aleatoria. Cada día un shock normal.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS
    z = rng.standard_normal((n_days, n_paths))
    daily_log_rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(daily_log_rets, axis=0)])
    return s0 * np.exp(log_paths)


# ---------------------------------------------------------------------------
# Bootstrap histórico: remuestrear días reales en vez de asumir normalidad
# ---------------------------------------------------------------------------

def simulate_bootstrap(
    s0: float,
    log_rets: pd.Series,
    n_days: int = TRADING_DAYS,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Cada día futuro se elige al azar (con reemplazo) entre los días reales del
    histórico. Conserva las colas gordas que la normal ignora, pero pierde el
    clustering de volatilidad (los días se barajan sin orden).
    """
    rng = np.random.default_rng(seed)
    sample = rng.choice(log_rets.to_numpy(), size=(n_days, n_paths), replace=True)
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(sample, axis=0)])
    return s0 * np.exp(log_paths)


# ---------------------------------------------------------------------------
# GBM correlacionado para varios activos
# ---------------------------------------------------------------------------

def simulate_correlated_gbm(
    s0: np.ndarray,
    mu: np.ndarray,
    cov_daily: np.ndarray,
    n_days: int = TRADING_DAYS,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Varios activos a la vez, respetando sus correlaciones históricas.

    Truco (Cholesky): generamos ruido independiente Z y lo multiplicamos por
    una matriz L tal que L @ L.T = covarianza. El resultado L @ Z tiene
    exactamente la covarianza que queremos.

    - s0, mu: arrays de longitud n_assets (mu ANUAL).
    - cov_daily: matriz de covarianza de los log-retornos DIARIOS.
    Devuelve array (n_days + 1, n_paths, n_assets).
    """
    rng = np.random.default_rng(seed)
    n_assets = len(s0)
    L = np.linalg.cholesky(cov_daily)
    z = rng.standard_normal((n_days, n_paths, n_assets))
    shocks = z @ L.T  # ahora cada día tiene la covarianza histórica
    daily_var = np.diag(cov_daily)
    drift = (mu / TRADING_DAYS - 0.5 * daily_var)
    daily_log_rets = drift + shocks
    log_paths = np.concatenate(
        [np.zeros((1, n_paths, n_assets)), np.cumsum(daily_log_rets, axis=0)], axis=0
    )
    return s0 * np.exp(log_paths)


# ---------------------------------------------------------------------------
# Medidas de riesgo de cola sobre la distribución simulada
# ---------------------------------------------------------------------------

def value_at_risk(final_returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    VaR al 95%: la pérdida que NO se supera en el 95% de los escenarios.
    Es el percentil 5 de los retornos (devuelto como número negativo).
    """
    return float(np.percentile(final_returns, (1 - confidence) * 100))


def conditional_var(final_returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    CVaR (Expected Shortfall): pérdida media en el peor 5% de escenarios.
    Responde "si las cosas van mal, ¿cuánto de mal?", que el VaR no dice.
    """
    var = value_at_risk(final_returns, confidence)
    tail = final_returns[final_returns <= var]
    return float(tail.mean())
