import numpy as np
import pandas as pd
from quantlab import timeseries as ts


def test_acf_of_white_noise_is_near_zero():
    x = np.random.default_rng(0).standard_normal(5000)
    assert np.all(np.abs(ts.acf(x, 10)) < 0.05)


def test_acf_of_ar1_matches_phi():
    rng = np.random.default_rng(1)
    x = np.zeros(20000)
    for t in range(1, len(x)):
        x[t] = 0.6 * x[t - 1] + rng.standard_normal()
    a = ts.acf(x, 3)
    assert np.allclose(a, [0.6, 0.36, 0.216], atol=0.03)


def test_garch_fit_recovers_parameters():
    omega, alpha, beta = 2e-6, 0.10, 0.85
    r = ts.simulate_garch(omega, alpha, beta, n_days=6000, n_paths=1, seed=3)[:, 0]
    fit = ts.garch_fit(pd.Series(r))
    assert abs(fit["alpha"] - alpha) < 0.05
    assert abs(fit["beta"] - beta) < 0.08
    assert fit["persistence"] < 1


def test_garch_simulation_has_fat_tails_and_clustering():
    r = ts.simulate_garch(2e-6, 0.12, 0.85, n_days=20000, seed=5)[:, 0]
    s = pd.Series(r)
    assert s.kurtosis() > 0.5                      # colas gordas
    assert ts.acf(r**2, 1)[0] > 0.1                # clustering: r² autocorrelado
    assert abs(ts.acf(r, 1)[0]) < 0.05             # dirección no predecible


def test_forecast_converges_to_long_run():
    r = ts.simulate_garch(2e-6, 0.10, 0.85, n_days=3000, seed=7)[:, 0]
    fit = ts.garch_fit(pd.Series(r))
    fc = ts.garch_forecast(fit, horizon=500)
    assert abs(fc[-1] - fit["long_run_vol"]) < 0.01


def test_regime_labels():
    r = ts.simulate_garch(2e-6, 0.10, 0.85, n_days=2000, seed=9)[:, 0]
    fit = ts.garch_fit(pd.Series(r))
    reg = ts.volatility_regime(fit)
    assert reg["regime"] in {"bajo", "normal", "alto"}
    assert 0 <= reg["percentile"] <= 100
