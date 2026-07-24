"""Synthetic RAPM recovery: simulate possessions from known player values,
assert the ridge recovers them. Catches sign errors, design-matrix bugs,
and prior-shrinkage math — the failure modes that matter."""
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rapm_lib as rl


def _simulate(n_players=40, n_poss=60000, seed=7):
    rng = np.random.default_rng(seed)
    true_o = rng.normal(0, 2.0, n_players)   # per-100 offensive value
    true_d = rng.normal(0, 1.5, n_players)   # per-100 points PREVENTED
    rows = []
    for _ in range(n_poss):
        pids = rng.choice(n_players, 10, replace=False)
        off, deff = pids[:5], pids[5:]
        hca = int(rng.integers(0, 2))
        mu = 1.10 + hca * 0.02 + (true_o[off].sum() - true_d[deff].sum()) / 100
        pts = rng.poisson(max(mu, 0.05))
        rows.append((*(100 + off), *(100 + deff), hca, pts))
    df = pd.DataFrame(rows, columns=["o1", "o2", "o3", "o4", "o5",
                                     "d1", "d2", "d3", "d4", "d5",
                                     "home_off", "pts"])
    return df, true_o, true_d


def test_ridge_exact_recovery_noiseless():
    """With deterministic y the algebra must be exact: estimates equal the
    true values up to the per-block mean absorbed by the intercept (ridge
    identifiability). Catches sign and design bugs with zero tolerance for
    luck."""
    rng = np.random.default_rng(2)
    n_players = 30
    true_o = rng.normal(0, 2.0, n_players)
    true_d = rng.normal(0, 1.5, n_players)
    rows = []
    for _ in range(20000):
        pids = rng.choice(n_players, 10, replace=False)
        off, deff = pids[:5], pids[5:]
        hca = int(rng.integers(0, 2))
        y100 = 110.0 + 2.0 * hca + true_o[off].sum() - true_d[deff].sum()
        rows.append((*(100 + off), *(100 + deff), hca, y100 / 100))
    df = pd.DataFrame(rows, columns=["o1", "o2", "o3", "o4", "o5",
                                     "d1", "d2", "d3", "d4", "d5",
                                     "home_off", "pts"])
    X, y, w, cols, _ = rl.build_design(df)
    pen = np.array([0.0 if c in ("_int", "_hca") else 1.0 for c in cols])
    b = rl.solve_ridge(X, y, w, lam=1e-6, penalize=pen)
    est_o = np.array([b[cols.index(f"o_{100+i}")] for i in range(n_players)])
    # -1 defender coding: a good defender's raw coefficient is already
    # positive (points prevented). No flip.
    est_d = np.array([b[cols.index(f"d_{100+i}")] for i in range(n_players)])
    assert np.max(np.abs((est_o - est_o.mean()) - (true_o - true_o.mean()))) < 0.05
    assert np.max(np.abs((est_d - est_d.mean()) - (true_d - true_d.mean()))) < 0.05
    assert abs(b[cols.index("_hca")] - 2.0) < 0.05


def test_ridge_recovers_synthetic_values():
    """Noisy (Poisson) simulation: recovery is noise-limited, thresholds
    are regression tripwires for this exact n/seed, not aspirations."""
    df, true_o, true_d = _simulate()
    X, y, w, cols, _games = rl.build_design(df)
    pen = np.array([0.0 if c in ("_int", "_hca") else 1.0 for c in cols])
    b = rl.solve_ridge(X, y, w, lam=300.0, penalize=pen)
    est_o = np.array([b[cols.index(f"o_{100+i}")] for i in range(40)])
    est_d = np.array([b[cols.index(f"d_{100+i}")] for i in range(40)])
    assert np.corrcoef(true_o, est_o)[0, 1] > 0.8
    assert np.corrcoef(true_d, est_d)[0, 1] > 0.65
    # unbiasedness: regression slope of est on true near 1 (not shrunk away)
    assert 0.85 < np.polyfit(true_o, est_o, 1)[0] < 1.2


def test_prior_shrinkage_pulls_toward_prior():
    df, true_o, _ = _simulate(n_poss=4000, seed=11)   # thin data
    X, y, w, cols, _games = rl.build_design(df)
    pen = np.array([0.0 if c in ("_int", "_hca") else 1.0 for c in cols])
    prior = np.zeros(len(cols))
    oracle = {f"o_{100+i}": true_o[i] for i in range(40)}
    for j, c in enumerate(cols):
        prior[j] = oracle.get(c, 0.0)
    b0 = rl.solve_ridge(X, y, w, 5000.0, pen)
    b1 = rl.solve_ridge(X, y, w, 5000.0, pen, prior=prior)
    est0 = np.array([b0[cols.index(f"o_{100+i}")] for i in range(40)])
    est1 = np.array([b1[cols.index(f"o_{100+i}")] for i in range(40)])
    r0 = np.corrcoef(true_o, est0)[0, 1]
    r1 = np.corrcoef(true_o, est1)[0, 1]
    assert r1 > r0   # an informative prior must help on thin data


def test_design_grouping_matches_ungrouped():
    """Grouped rows (weight = n_poss, y = mean) must produce the same
    normal equations as one row per possession."""
    df, _, _ = _simulate(n_poss=3000, seed=3)
    # duplicate every possession row 2x: grouping should collapse them
    df2 = pd.concat([df, df], ignore_index=True)
    X1, y1, w1, cols1, _ = rl.build_design(df)
    X2, y2, w2, cols2, _ = rl.build_design(df2)
    assert cols1 == cols2
    pen = np.array([0.0 if c in ("_int", "_hca") else 1.0 for c in cols1])
    b1 = rl.solve_ridge(X1, y1, w1, 100.0, pen)
    b2 = rl.solve_ridge(X2, y2, w2, 200.0, pen)  # 2x data -> 2x lambda
    assert np.allclose(b1, b2, atol=1e-8)


def test_ridge_se_shape_and_positivity():
    df, _, _ = _simulate(n_poss=8000, seed=5)
    X, y, w, cols, _ = rl.build_design(df)
    pen = np.array([0.0 if c in ("_int", "_hca") else 1.0 for c in cols])
    b = rl.solve_ridge(X, y, w, 300.0, pen)
    se = rl.ridge_se(X, y, w, 300.0, pen, b)
    assert se.shape == b.shape
    assert (se[2:] > 0).all()


if __name__ == "__main__":
    for f in [test_ridge_exact_recovery_noiseless,
              test_ridge_recovers_synthetic_values,
              test_prior_shrinkage_pulls_toward_prior,
              test_design_grouping_matches_ungrouped,
              test_ridge_se_shape_and_positivity]:
        f()
        print(f"ok {f.__name__}")
