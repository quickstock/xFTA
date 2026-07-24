"""WO-8 — is a minutes-played + team-strength prior better than the box-score one?

The literature reports that priors built from minutes played and team strength are
more reliable than age-based priors. The prior shipped here is neither: it regresses
other seasons' plain RAPM on per-100 box-score rates. So the open question is which
of those two wins, and the spec asks for the answer on out-of-sample game-margin
RMSE.

The instrument is `cv_rmse(prior=...)`, which sets the player coefficients to the
prior and refits only the intercept and home-court term, then scores held-out game
margins. That is exactly "how well does this prior alone predict games it has not
seen" — the sum-of-parts baseline — so a straight comparison between two priors is
well defined. Lower is better.

Both priors are built leave-one-season-out and both get the same sample-size
shrinkage toward the fitted league mean, so the comparison isolates the feature set
rather than the fitting scheme.

Read-only: this script decides which prior wins. It does not overwrite the shipped
model.
"""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import rapm_lib as rl
from train_rapm import (BOX_DIR, BOX_FEATURES, PRIOR_SHRINK_POSS, cv_rmse,
                        fold_of, gram_parts)
from export_rapm import TEAM_ABBREV
from scipy import linalg
from train_rapm import N_FOLDS, game_margins


def cv_rmse_prior_ridge(X, y, w, game_ids, folds, sign, G_tot, v_tot, parts,
                        pen, lam, prior):
    """Held-out game-margin RMSE for the ridge fit SHRINKING TOWARD `prior`.

    cv_rmse(prior=...) scores the prior alone (sum of parts). This scores what the
    model actually does with it: solve the penalised system for the deviation from
    the prior, exactly as rapm_lib.solve_ridge(prior=...) does. Which prior makes a
    better shrinkage target is the question that decides what ships."""
    rmses = []
    for f in range(N_FOLDS):
        G_f, v_f = parts[f]
        G_tr, v_tr = G_tot - G_f, v_tot - v_f
        A = G_tr + lam * np.diag(pen)
        b = prior + linalg.solve(A, v_tr - G_tr @ prior, assume_a="pos")
        m = folds == f
        pred, act = game_margins(X[m], y[m], w[m], game_ids[m], sign[m], b)
        rmses.append(float(np.sqrt(np.mean((pred - act) ** 2))))
    return float(np.mean(rmses))

MINUTES_FEATURES = ["MIN", "MIN_PER_GAME", "TEAM_NET100"]


def team_net_ratings(con):
    """Net points per 100 possessions per (season, team abbrev), from the
    possession stream — the natural team-strength measure for this project."""
    pl = pd.read_sql(
        "SELECT game_id, season, off_team, pts FROM possession_lineups", con)
    games = pd.read_sql(
        "SELECT GAME_ID game_id, home_team_id, away_team_id FROM games", con)
    pl = pl.merge(games, on="game_id", how="left")
    pl["def_team"] = pl.away_team_id.where(
        pl.off_team == pl.home_team_id, pl.home_team_id)
    pl = pl[(pl.off_team > 0) & (pl.def_team > 0)]

    off = pl.groupby(["season", "off_team"]).pts.agg(["sum", "size"])
    dfn = pl.groupby(["season", "def_team"]).pts.agg(["sum", "size"])
    off.index.names = dfn.index.names = ["season", "team"]
    j = off.join(dfn, lsuffix="_off", rsuffix="_def", how="inner").reset_index()
    j["TEAM_NET100"] = 100 * (j["sum_off"] / j["size_off"]
                              - j["sum_def"] / j["size_def"])
    j["TEAM_ABBREVIATION"] = j.team.map(
        lambda t: TEAM_ABBREV.get(int(t), "UNK"))
    return j[["season", "TEAM_ABBREVIATION", "TEAM_NET100"]]


def build_prior(rapm_by_season, feat_by_season, features, target_season,
                cols_pids, poss_by_pid):
    """LOSO prior over an arbitrary feature set, with the same sample-size
    shrinkage the shipped prior uses."""
    train = []
    for s, tab in rapm_by_season.items():
        if s == target_season or s == "pooled":
            continue
        f = feat_by_season.get(s)
        if f is None:
            continue
        m = tab.merge(f, left_on="player_id", right_on="PLAYER_ID")
        train.append(m[m.poss_off + m.poss_def >= 1000])
    if not train:
        return {}, None, None
    tr = pd.concat(train, ignore_index=True).dropna(subset=features)
    Xb = tr[features].to_numpy(dtype=float)
    mu, sd = Xb.mean(axis=0), Xb.std(axis=0) + 1e-9
    Xb = np.column_stack([np.ones(len(Xb)), (Xb - mu) / sd])

    def fit(yv):
        beta, *_ = np.linalg.lstsq(Xb, yv, rcond=None)
        pred = Xb @ beta
        r2 = 1 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2)
        return beta, float(r2)

    beta_o, r2_o = fit(tr.o.to_numpy())
    beta_d, r2_d = fit(tr.d.to_numpy())

    ft = feat_by_season.get(target_season)
    prior = {}
    if ft is not None:
        ft = ft.dropna(subset=features)
        Xt = ft[features].to_numpy(dtype=float)
        Xt = np.column_stack([np.ones(len(Xt)), (Xt - mu) / sd])
        po, pd_ = Xt @ beta_o, Xt @ beta_d
        base_o, base_d = float(beta_o[0]), float(beta_d[0])
        for pid, o_hat, d_hat in zip(ft.PLAYER_ID, po, pd_):
            p = int(pid)
            if p not in cols_pids:
                continue
            n_o, n_d = poss_by_pid.get(p, (0.0, 0.0))
            w_o = n_o / (n_o + PRIOR_SHRINK_POSS)
            w_d = n_d / (n_d + PRIOR_SHRINK_POSS)
            prior[f"o_{p}"] = base_o + w_o * (float(o_hat) - base_o)
            prior[f"d_{p}"] = base_d + w_d * (float(d_hat) - base_d)
    return prior, round(r2_o, 3), round(r2_d, 3)


def main():
    con = sqlite3.connect(config.DB_PATH)
    pl = pd.read_sql("SELECT * FROM possession_lineups", con)
    rapm = pd.read_sql("SELECT * FROM rapm WHERE season != 'pooled'", con)
    nets = team_net_ratings(con)

    seasons = sorted(pl.season.unique())
    box, mins = {}, {}
    for s in seasons:
        p = os.path.join(BOX_DIR, f"{s}.parquet")
        if not os.path.exists(p):
            continue
        b = pd.read_parquet(p)
        box[s] = b
        m = b[["PLAYER_ID", "TEAM_ABBREVIATION", "MIN", "GP"]].copy()
        m["MIN_PER_GAME"] = m.MIN / m.GP.replace(0, np.nan)
        m = m.merge(nets[nets.season == s].drop(columns=["season"]),
                    on="TEAM_ABBREVIATION", how="left")
        mins[s] = m

    rapm_by_season = {s: g for s, g in rapm.groupby("season")}

    print("=== WO-8: prior comparison, out-of-sample game-margin RMSE ===")
    print("(cv_rmse with the prior as coefficients; intercept + HCA refit only)")
    print(f"{'season':<9}{'box RMSE':>10}{'min+team':>10}{'winner':>10}"
          f"{'box R2 o/d':>14}{'min R2 o/d':>14}")
    rows = []
    for s in seasons:
        if s not in mins:
            continue
        df = pl[pl.season == s]
        X, y, w, cols, game_ids = rl.build_design(df)
        pen = np.array([0.0 if c in ("_int", "_hca") else 1.0 for c in cols])
        folds = fold_of(game_ids)
        key = df.groupby(["game_id", "o1", "o2", "o3", "o4", "o5",
                          "d1", "d2", "d3", "d4", "d5", "home_off"],
                         sort=False).size().reset_index()
        sign = np.where(key.home_off.to_numpy() == 1, 1.0, -1.0)
        G_tot, v_tot, parts = gram_parts(X, y, w, folds)

        tab = rapm_by_season[s]
        pids = set(tab.player_id)
        poss = {int(p): (float(a), float(b)) for p, a, b in
                zip(tab.player_id, tab.poss_off, tab.poss_def)}

        pm_box, r2bo, r2bd = build_prior(rapm_by_season, box, BOX_FEATURES,
                                         s, pids, poss)
        pm_min, r2mo, r2md = build_prior(rapm_by_season, mins,
                                         MINUTES_FEATURES, s, pids, poss)

        def rmse_of(pm):
            if not pm:
                return None
            v = np.zeros(len(cols))
            for j, c in enumerate(cols):
                v[j] = pm.get(c, 0.0)
            return cv_rmse(X, y, w, game_ids, folds, sign, G_tot, v_tot,
                           parts, pen, prior=v)

        rb, rm = rmse_of(pm_box), rmse_of(pm_min)

        # the decisive comparison: each prior as a SHRINKAGE TARGET at the
        # shipped lambda, not as a standalone predictor
        LAM = 3200.0
        def ridge_of(pm):
            v = np.zeros(len(cols))
            for j, c in enumerate(cols):
                v[j] = pm.get(c, 0.0)
            return cv_rmse_prior_ridge(X, y, w, game_ids, folds, sign,
                                       G_tot, v_tot, parts, pen, LAM, v)
        gb, gm = ridge_of(pm_box), ridge_of(pm_min)
        print(f"{'':<9}{gb:>10.4f}{gm:>10.4f}"
              f"{('box' if gb < gm else 'min+team'):>10}   <- as shrinkage target")
        win = ("box" if rb is not None and (rm is None or rb < rm)
               else "min+team")
        print(f"{s:<9}{rb:>10.4f}{rm:>10.4f}{win:>10}"
              f"{f'{r2bo}/{r2bd}':>14}{f'{r2mo}/{r2md}':>14}")
        rows.append({"season": s, "boxRmse": round(rb, 5),
                     "minutesTeamRmse": round(rm, 5), "winner": win,
                     "boxRidgeRmse": round(gb, 5),
                     "minutesTeamRidgeRmse": round(gm, 5),
                     "ridgeWinner": "box" if gb < gm else "min+team",
                     "boxR2": {"o": r2bo, "d": r2bd},
                     "minutesTeamR2": {"o": r2mo, "d": r2md}})
    con.close()

    box_ridge_wins = sum(r["ridgeWinner"] == "box" for r in rows)
    mean_gb = float(np.mean([r["boxRidgeRmse"] for r in rows]))
    mean_gm = float(np.mean([r["minutesTeamRidgeRmse"] for r in rows]))
    print(f"\n  AS SHRINKAGE TARGET (what ships): box {mean_gb:.4f} vs "
          f"minutes+team {mean_gm:.4f} (diff {mean_gm - mean_gb:+.4f})")
    print(f"  box wins {box_ridge_wins}/{len(rows)} seasons as shrinkage target")
    box_wins = sum(r["winner"] == "box" for r in rows)
    mean_box = float(np.mean([r["boxRmse"] for r in rows]))
    mean_min = float(np.mean([r["minutesTeamRmse"] for r in rows]))
    print(f"\n  mean RMSE: box {mean_box:.4f} vs minutes+team {mean_min:.4f} "
          f"(diff {mean_min - mean_box:+.4f})")
    print(f"  box prior wins {box_wins}/{len(rows)} seasons")
    # the shrinkage-target comparison decides, since that is how the prior is used
    verdict = ("keep the box-score prior"
               if box_ridge_wins > len(rows) / 2
               else "switch to the minutes + team-strength prior")
    print(f"  VERDICT: {verdict}")

    out = {"perSeason": rows, "meanBoxRmse": round(mean_box, 5),
           "meanMinutesTeamRmse": round(mean_min, 5),
           "boxWins": box_wins, "nSeasons": len(rows),
           "meanBoxRidgeRmse": round(mean_gb, 5),
           "meanMinutesTeamRidgeRmse": round(mean_gm, 5),
           "boxRidgeWins": box_ridge_wins,
           "decidedBy": ("shrinkage-target RMSE, because that is how the prior "
                         "is actually used; the standalone comparison is "
                         "reported alongside"),
           "features": {"box": BOX_FEATURES, "minutesTeam": MINUTES_FEATURES},
           "instrument": ("cv_rmse with the prior as player coefficients and "
                          "only intercept + home-court refit; held-out "
                          "game-margin RMSE averaged over 10 folds"),
           "verdict": verdict}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "model_artifacts", "wo8_prior_variant.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("  wrote model_artifacts/wo8_prior_variant.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
