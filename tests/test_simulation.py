import numpy as np
import pandas as pd
from quantlab import simulation as sim


def test_gbm_shape_and_start():
    paths = sim.simulate_gbm(100, 0.08, 0.2, n_days=252, n_paths=500)
    assert paths.shape == (253, 500)
    assert np.all(paths[0] == 100)


def test_gbm_mean_matches_theory():
    # E[S_T] = S0 * exp(mu * T)
    paths = sim.simulate_gbm(100, 0.08, 0.2, n_days=252, n_paths=200_000, seed=0)
    assert np.isclose(paths[-1].mean(), 100 * np.exp(0.08), rtol=0.01)


def test_correlated_gbm_recovers_correlation():
    corr = np.array([[1.0, 0.7], [0.7, 1.0]])
    vols = np.array([0.2, 0.3]) / np.sqrt(252)
    cov = np.outer(vols, vols) * corr
    paths = sim.simulate_correlated_gbm(np.array([100.0, 50.0]), np.array([0.05, 0.05]),
                                        cov, n_days=252, n_paths=5000, seed=1)
    lr = np.diff(np.log(paths), axis=0).reshape(-1, 2)
    assert np.isclose(np.corrcoef(lr.T)[0, 1], 0.7, atol=0.02)


def test_var_and_cvar_known_values():
    r = np.linspace(-0.5, 0.5, 1001)  # uniforme de -50% a +50%
    assert np.isclose(sim.value_at_risk(r, 0.95), -0.45, atol=0.001)
    assert sim.conditional_var(r, 0.95) < sim.value_at_risk(r, 0.95)


def test_sortino_penalizes_only_downside():
    from quantlab import metrics as m
    up = pd.Series([0.05, 0.0, 0.05, 0.0, -0.01] * 100)
    assert m.sortino_ratio(up) > m.sharpe_ratio(up)
