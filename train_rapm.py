"""O/D RAPM: weighted ridge on possession lineups (template: Sill 2010).

Per season:
  - grouped design (game_id in the group key so CV slices by game);
  - lambda by 10-fold GAME-holdout CV on out-of-sample game-margin RMSE,
    reported against two baselines on the same folds: intercept+HCA only,
    and the box prior summed with no per-player refit;
  - plain fit at the chosen lambda (d is positive-is-good under the -1
    defender coding, no flip anywhere);
  - a leave-one-season-out box prior: other seasons' plain RAPM regressed
    on their per-100 box stats, predicted onto this season's players, then
    the ridge refit shrinking toward it (the default display variant);
  - sandwich SEs on the plain fit;
  - near-inseparable same-team pairs (indicator corr > 0.95, >= 300 shared
    possessions).
Plus one pooled all-seasons fit (season = "pooled").

Outputs: rapm table + rapm_meta (single JSON row: lambdas, CV table,
collinear pairs, prior fit R^2, skip stats).
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd
from scipy import linalg

import config
import rapm_lib as rl

# y is per-100 points with large per-possession variance, so the optimal
# ridge penalty sits well above the classic per-possession-scale references
# (lambda_min~241, lambda_1SE~1068 in Sill's units); the grid runs high
# enough to bracket a true interior minimum rather than pinning a boundary.
LAMBDA_GRID = [200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0,
               12800.0, 25600.0, 51200.0]
N_FOLDS = 10
BOX_FEATURES = ["PTS", "FGA", "FG3A", "FTA", "OREB", "DREB", "AST",
                "TOV", "STL", "BLK", "PF"]
BOX_DIR = os.path.join(config.CACHE_DIR, "box100")


def fold_of(game_ids: np.ndarray) -> np.ndarray:
    """Deterministic game -> fold assignment (hash of the id string)."""
    import zlib
    uniq = {g: zlib.crc32(str(g).encode()) % N_FOLDS
            for g in pd.unique(game_ids)}
    return np.array([uniq[g] for g in game_ids])


def gram_parts(X, y, w, folds):
    """Per-fold Gram partials: X'WX and X'Wy are additive over rows, so
    each (fold, lambda) fit is a subtraction plus one dense solve."""
    G_tot = (X.T @ X.multiply(w[:, None])).toarray()
    v_tot = X.T @ (w * y)
    parts = {}
    for f in range(N_FOLDS):
        m = folds == f
        Xf = X[m]
        wf, yf = w[m], y[m]
        parts[f] = ((Xf.T @ Xf.multiply(wf[:, None])).toarray(),
                    Xf.T @ (wf * yf))
    return G_tot, v_tot, parts


def game_margins(X, y, w, game_ids, home_off_sign, b):
    """Per-game predicted and actual margins (home minus away, points)."""
    pred_rows = np.asarray(X @ b).ravel()
    dfm = pd.DataFrame({
        "game": game_ids,
        "pred": home_off_sign * pred_rows * w / 100.0,
        "act": home_off_sign * y * w / 100.0,
    })
    g = dfm.groupby("game").sum()
    return g.pred.to_numpy(), g.act.to_numpy()


def cv_rmse(X, y, w, game_ids, folds, sign, G_tot, v_tot, parts,
            pen, lam=None, prior=None, hca_only=False):
    """Mean over folds of held-out game-margin RMSE for one model spec."""
    rmses = []
    for f in range(N_FOLDS):
        G_f, v_f = parts[f]
        G_tr, v_tr = G_tot - G_f, v_tot - v_f
        if hca_only:
            b = np.zeros(len(pen))
            A2 = G_tr[:2, :2]
            b[:2] = linalg.solve(A2, v_tr[:2], assume_a="pos")
        elif prior is not None:
            b = prior.copy()
            rhs = v_tr - G_tr @ prior
            A2 = G_tr[:2, :2]
            b[:2] += linalg.solve(A2, rhs[:2], assume_a="pos")
        else:
            A = G_tr + lam * np.diag(pen)
            b = linalg.solve(A, v_tr, assume_a="pos")
        m = folds == f
        pred, act = game_margins(X[m], y[m], w[m], game_ids[m], sign[m], b)
        rmses.append(float(np.sqrt(np.mean((pred - act) ** 2))))
    return float(np.mean(rmses))


def fit_season(df, prior_map=None):
    """Returns (result dict). df = possession_lineups rows of one scope."""
    X, y, w, cols, game_ids = rl.build_design(df)
    pen = np.array([0.0 if c in ("_int", "_hca") else 1.0 for c in cols])
    folds = fold_of(game_ids)
    # +1 when the offense on a row is the home team, else -1
    key = df.groupby(["game_id", "o1", "o2", "o3", "o4", "o5",
                      "d1", "d2", "d3", "d4", "d5", "home_off"],
                     sort=False).size().reset_index()
    sign = np.where(key.home_off.to_numpy() == 1, 1.0, -1.0)
    G_tot, v_tot, parts = gram_parts(X, y, w, folds)

    cv = {lam: cv_rmse(X, y, w, game_ids, folds, sign, G_tot, v_tot,
                       parts, pen, lam=lam) for lam in LAMBDA_GRID}
    lam_star = min(cv, key=cv.get)
    rmse_hca = cv_rmse(X, y, w, game_ids, folds, sign, G_tot, v_tot,
                       parts, pen, hca_only=True)

    prior = np.zeros(len(cols))
    rmse_prior_sum = None
    if prior_map:
        for j, c in enumerate(cols):
            prior[j] = prior_map.get(c, 0.0)
        rmse_prior_sum = cv_rmse(X, y, w, game_ids, folds, sign, G_tot,
                                 v_tot, parts, pen, prior=prior)

    b_plain = rl.solve_ridge(X, y, w, lam_star, pen)
    se = rl.ridge_se(X, y, w, lam_star, pen, b_plain)
    b_prior = (rl.solve_ridge(X, y, w, lam_star, pen, prior=prior)
               if prior_map else b_plain)

    # exposure counts
    poss_off, poss_def = {}, {}
    for slot in ("o1", "o2", "o3", "o4", "o5"):
        for pid, n in df.groupby(slot).size().items():
            poss_off[pid] = poss_off.get(pid, 0) + int(n)
    for slot in ("d1", "d2", "d3", "d4", "d5"):
        for pid, n in df.groupby(slot).size().items():
            poss_def[pid] = poss_def.get(pid, 0) + int(n)

    rows = []
    for j, c in enumerate(cols):
        if c.startswith("o_"):
            pid = int(c[2:])
            jd = cols.index(f"d_{pid}") if f"d_{pid}" in cols else None
            rows.append({
                "player_id": pid,
                "poss_off": poss_off.get(pid, 0),
                "poss_def": poss_def.get(pid, 0),
                "o": float(b_plain[j]),
                "d": float(b_plain[jd]) if jd is not None else 0.0,
                "o_p": float(b_prior[j]),
                "d_p": float(b_prior[jd]) if jd is not None else 0.0,
                "se_o": float(se[j]),
                "se_d": float(se[jd]) if jd is not None else 0.0,
            })
    out = pd.DataFrame(rows)
    out["net"] = out.o + out.d
    out["net_p"] = out.o_p + out.d_p
    return {
        "table": out,
        "lambda": lam_star,
        "cv": {str(k): round(v, 4) for k, v in cv.items()},
        "rmse_rapm": round(cv[lam_star], 4),
        "rmse_hca": round(rmse_hca, 4),
        "rmse_prior_sum": (round(rmse_prior_sum, 4)
                           if rmse_prior_sum is not None else None),
        "anchor": float(np.average(np.asarray(X @ b_prior).ravel(),
                                   weights=w) - np.average(y, weights=w)),
    }


def collinear_pairs(df, season):
    """Same-team pairs whose on-court indicators are near-duplicates."""
    out = []
    team_of = {}
    for slot in ("o1", "o2", "o3", "o4", "o5"):
        for (pid, team), n in df.groupby([slot, "off_team"]).size().items():
            team_of.setdefault(pid, {})
            team_of[pid][team] = team_of[pid].get(team, 0) + int(n)
    main_team = {p: max(t, key=t.get) for p, t in team_of.items()}

    for team in sorted(set(main_team.values())):
        pids = sorted(p for p, t in main_team.items() if t == team)
        if len(pids) < 2:
            continue
        sub = df[(df.off_team == team)]
        poss_idx = pd.RangeIndex(len(sub))
        on = {}
        arr = sub[["o1", "o2", "o3", "o4", "o5"]].to_numpy()
        for p in pids:
            on[p] = (arr == p).any(axis=1).astype(float)
        for i, a in enumerate(pids):
            for b in pids[i + 1:]:
                shared = float((on[a] * on[b]).sum())
                if shared < 300:
                    continue
                r = float(np.corrcoef(on[a], on[b])[0, 1])
                if r > 0.95:
                    out.append({"season": season, "a": str(a), "b": str(b),
                                "r": round(r, 3),
                                "sharedPoss": int(shared)})
    return out


def build_prior_map(rapm_by_season, box_by_season, target_season, cols_pids):
    """LOSO: regress other seasons' plain o (resp. d) on standardized box
    per-100 stats, predict this season's players. Returns ({col: prior},
    r2_o, r2_d)."""
    train_rows = []
    for s, tab in rapm_by_season.items():
        if s == target_season or s == "pooled":
            continue
        box = box_by_season.get(s)
        if box is None:
            continue
        merged = tab.merge(box, left_on="player_id", right_on="PLAYER_ID")
        merged = merged[merged.poss_off + merged.poss_def >= 1000]
        train_rows.append(merged)
    if not train_rows:
        return {}, None, None
    tr = pd.concat(train_rows, ignore_index=True)
    Xb = tr[BOX_FEATURES].to_numpy(dtype=float)
    mu, sd = Xb.mean(axis=0), Xb.std(axis=0) + 1e-9
    Xb = np.column_stack([np.ones(len(Xb)), (Xb - mu) / sd])

    def fit(yv):
        beta, *_ = np.linalg.lstsq(Xb, yv, rcond=None)
        pred = Xb @ beta
        ss = 1 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2)
        return beta, float(ss)

    beta_o, r2_o = fit(tr.o.to_numpy())
    beta_d, r2_d = fit(tr.d.to_numpy())

    box_t = box_by_season.get(target_season)
    prior = {}
    if box_t is not None:
        Xt = box_t[BOX_FEATURES].to_numpy(dtype=float)
        Xt = np.column_stack([np.ones(len(Xt)), (Xt - mu) / sd])
        po = Xt @ beta_o
        pd_ = Xt @ beta_d
        for pid, o_hat, d_hat in zip(box_t.PLAYER_ID, po, pd_):
            if int(pid) in cols_pids:
                prior[f"o_{int(pid)}"] = float(o_hat)
                prior[f"d_{int(pid)}"] = float(d_hat)
    return prior, round(r2_o, 3), round(r2_d, 3)


def main():
    con = sqlite3.connect(config.DB_PATH)
    pl = pd.read_sql("SELECT * FROM possession_lineups", con)
    skips = pd.read_sql("SELECT reason, COUNT(*) n FROM stint_skips "
                        "GROUP BY reason", con)
    seasons = sorted(pl.season.unique())
    box_by_season = {}
    for s in seasons:
        p = os.path.join(BOX_DIR, f"{s}.parquet")
        if os.path.exists(p):
            box_by_season[s] = pd.read_parquet(p)

    # pass 1: plain fits (also the prior's training signal)
    print("pass 1: plain ridge per season")
    results = {}
    for s in seasons:
        results[s] = fit_season(pl[pl.season == s])
        print(f"  [{s}] lam={results[s]['lambda']:.0f} "
              f"rmse={results[s]['rmse_rapm']} vs hca {results[s]['rmse_hca']}",
              flush=True)
    rapm_by_season = {s: r["table"] for s, r in results.items()}

    # pass 2: LOSO priors + refit
    print("pass 2: LOSO box prior + refit")
    meta_cv = []
    prior_r2 = {}
    for s in seasons:
        pids = set(rapm_by_season[s].player_id)
        prior_map, r2o, r2d = build_prior_map(
            rapm_by_season, box_by_season, s, pids)
        prior_r2[s] = {"o": r2o, "d": r2d}
        results[s] = fit_season(pl[pl.season == s], prior_map=prior_map)
        r = results[s]
        meta_cv.append({"season": s, "lam": r["lambda"],
                        "rmseRapm": r["rmse_rapm"],
                        "rmsePriorSum": r["rmse_prior_sum"],
                        "rmseHca": r["rmse_hca"]})
        print(f"  [{s}] lam={r['lambda']:.0f} rapm={r['rmse_rapm']} "
              f"prior-sum={r['rmse_prior_sum']} hca={r['rmse_hca']} "
              f"anchor={r['anchor']:+.3f}", flush=True)

    # pooled fit (no prior: the pool IS the information)
    print("pooled fit")
    pooled = fit_season(pl)
    print(f"  [pooled] lam={pooled['lambda']:.0f} rmse={pooled['rmse_rapm']}")

    # names
    names = {}
    for s, box in box_by_season.items():
        for pid, nm in zip(box.PLAYER_ID, box.PLAYER_NAME):
            names[int(pid)] = nm
    ps = pd.read_sql("SELECT DISTINCT player_id, player_name "
                     "FROM player_season", con)
    for pid, nm in zip(ps.player_id, ps.player_name):
        names.setdefault(int(pid), nm)

    frames = []
    for s in seasons:
        t = results[s]["table"].copy()
        t["season"] = s
        frames.append(t)
    tp = pooled["table"].copy()
    tp["season"] = "pooled"
    frames.append(tp)
    rapm = pd.concat(frames, ignore_index=True)
    rapm["player_name"] = rapm.player_id.map(names).fillna(
        rapm.player_id.astype(str))
    for c in ("o", "d", "net", "o_p", "d_p", "net_p", "se_o", "se_d"):
        rapm[c] = rapm[c].round(3)
    rapm.to_sql("rapm", con, if_exists="replace", index=False)

    collinear = []
    for s in seasons:
        collinear.extend(collinear_pairs(pl[pl.season == s], s))
    for c in collinear:
        c["aName"] = names.get(int(c["a"]), c["a"])
        c["bName"] = names.get(int(c["b"]), c["b"])

    meta = {
        "lambda": {s: results[s]["lambda"] for s in seasons},
        "lambdaPooled": pooled["lambda"],
        "cv": meta_cv,
        "cvPooled": {"rmseRapm": pooled["rmse_rapm"],
                     "rmseHca": pooled["rmse_hca"]},
        "collinear": collinear,
        "boxPriorR2": prior_r2,
        "skips": {r.reason: int(r.n) for r in skips.itertuples()},
        "gamesKept": int(pl.game_id.nunique()),
    }
    pd.DataFrame({"json": [json.dumps(meta)]}).to_sql(
        "rapm_meta", con, if_exists="replace", index=False)
    con.commit()

    print(f"\nrapm: {len(rapm)} rows "
          f"({rapm.season.nunique()} seasons incl. pooled)")
    print(f"collinear pairs: {len(collinear)}")
    lb = rapm[(rapm.season == "pooled")
              & (rapm.poss_off + rapm.poss_def >= 12000)]
    print("\npooled top-10 by net (prior variant), laugh test:")
    print(lb.nlargest(10, "net_p")[
        ["player_name", "poss_off", "poss_def", "o_p", "d_p", "net_p"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
