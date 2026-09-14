"""
Tests con datos sintéticos donde la respuesta correcta se conoce a mano.
Correr: pytest
"""
import numpy as np
import pandas as pd
from quantlab import metrics as m


def test_simple_and_log_returns_agree_for_small_moves():
    prices = pd.Series([100, 101, 102, 101.5])
    s, l = m.simple_returns(prices), m.log_returns(prices)
    assert np.allclose(s, l, atol=1e-3)


def test_log_returns_sum_to_total():
    prices = pd.Series([100, 110, 99, 120])
    assert np.isclose(m.log_returns(prices).sum(), np.log(120 / 100))


def test_cumulative_returns_recover_price_path():
    prices = pd.Series([100.0, 110, 99, 120])
    wealth = m.cumulative_returns(m.simple_returns(prices))
    assert np.isclose(wealth.iloc[-1], 1.20)


def test_max_drawdown_known_case():
    # sube a 120, cae a 90 -> drawdown = 90/120 - 1 = -25%
    prices = pd.Series([100.0, 120, 90, 95])
    assert np.isclose(m.max_drawdown(m.simple_returns(prices)), -0.25)


def test_annualized_vol_scales_with_sqrt():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 5000))
    assert np.isclose(m.annualized_volatility(r), 0.01 * np.sqrt(252), rtol=0.05)


def test_summary_columns_and_shape():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=list("ABC"))
    s = m.summary(r)
    assert list(s.index) == list("ABC")
    assert {"ret_anual", "vol_anual", "sharpe", "max_drawdown"} <= set(s.columns)
