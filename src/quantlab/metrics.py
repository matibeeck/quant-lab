"""
Métricas básicas de retornos y riesgo.

Convención: todas las funciones aceptan un pd.Series (un activo) o un
pd.DataFrame (varios activos, una columna por activo) y devuelven el mismo
tipo. Los retornos son DIARIOS salvo que se diga lo contrario.

TRADING_DAYS = 252 porque un año tiene ~252 días hábiles de bolsa.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Retornos
# ---------------------------------------------------------------------------

def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """r_t = P_t / P_{t-1} - 1.  Lo que 'sientes' en la cuenta."""
    return prices.pct_change().dropna()


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """r_t = ln(P_t / P_{t-1}).  Se suman en el tiempo y se modelan mejor."""
    return np.log(prices / prices.shift(1)).dropna()


def cumulative_returns(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Crecimiento de 1 unidad invertida: prod(1 + r_t).  Usa retornos SIMPLES."""
    return (1 + returns).cumprod()


# ---------------------------------------------------------------------------
# Riesgo y rendimiento anualizados
# ---------------------------------------------------------------------------

def annualized_return(returns: pd.DataFrame | pd.Series, periods: int = TRADING_DAYS):
    """Retorno geométrico anual (CAGR) a partir de retornos simples diarios."""
    total_growth = (1 + returns).prod()
    n_years = len(returns) / periods
    return total_growth ** (1 / n_years) - 1


def annualized_volatility(returns: pd.DataFrame | pd.Series, periods: int = TRADING_DAYS):
    """std diaria * sqrt(252).  La raíz aparece porque la varianza es aditiva, no la std."""
    return returns.std() * np.sqrt(periods)


def rolling_volatility(returns: pd.DataFrame | pd.Series, window: int = 21, periods: int = TRADING_DAYS):
    """Volatilidad anualizada calculada sobre una ventana móvil (21 ≈ un mes hábil)."""
    return returns.rolling(window).std() * np.sqrt(periods)


def sharpe_ratio(returns: pd.DataFrame | pd.Series, risk_free_annual: float = 0.0, periods: int = TRADING_DAYS):
    """
    (retorno - tasa libre de riesgo) / volatilidad, anualizado.
    Mide cuánto retorno extra obtienes por unidad de riesgo.
    """
    rf_daily = risk_free_annual / periods
    excess = returns - rf_daily
    return excess.mean() / excess.std() * np.sqrt(periods)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def drawdown(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Caída desde el máximo histórico hasta hoy, en %.
    0 = estás en máximo; -0.30 = estás 30% por debajo del pico anterior.
    """
    wealth = cumulative_returns(returns)
    running_max = wealth.cummax()
    return wealth / running_max - 1


def max_drawdown(returns: pd.DataFrame | pd.Series):
    """La peor caída pico-a-valle del período."""
    return drawdown(returns).min()


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def summary(returns: pd.DataFrame, risk_free_annual: float = 0.0) -> pd.DataFrame:
    """Tabla con las métricas clave por activo. Espera retornos SIMPLES diarios."""
    return pd.DataFrame({
        "ret_anual": annualized_return(returns),
        "vol_anual": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free_annual),
        "max_drawdown": max_drawdown(returns),
        "skew": returns.skew(),
        "kurtosis": returns.kurtosis(),  # exceso de curtosis: normal = 0
    })


def sortino_ratio(returns: pd.DataFrame | pd.Series, risk_free_annual: float = 0.0, periods: int = TRADING_DAYS):
    """
    Como Sharpe, pero en el denominador solo cuenta la volatilidad de los días
    negativos. Idea: la volatilidad "buena" (subidas grandes) no debería penalizar.
    """
    rf_daily = risk_free_annual / periods
    excess = returns - rf_daily
    downside = excess.where(excess < 0, 0.0)
    downside_std = np.sqrt((downside**2).mean())
    return excess.mean() / downside_std * np.sqrt(periods)


def calmar_ratio(returns: pd.DataFrame | pd.Series, periods: int = TRADING_DAYS):
    """Retorno anual / |max drawdown|. Retorno por unidad de 'dolor máximo'."""
    return annualized_return(returns, periods) / max_drawdown(returns).abs()
