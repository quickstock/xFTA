"""Layer 3 foundation — in-game win probability. possessions +
possession_lineups -> winprob_states, winprob_meta.

Template is McFarlane (2019): P(win) = f(score margin, time remaining), fit by
logistic regression on possession-start states, split by game so no game
contributes to both fit and evaluation.

**No point-spread term.** The published model adds the closing spread as a team
strength proxy. There is no free, licensed source of historical NBA spreads for
these seasons, so this build uses the documented fallback: margin and time only,
with their interaction. The cost is real and worth stating — without a strength
prior, a tie between mismatched teams is called 50/50 — so the model is used for
*decision* comparisons within a single game state, where both branches share
whatever strength bias exists and it largely cancels, rather than for predicting
who wins.

Regulation states only. A game tied at the end of regulation is a coin flip this
model has no information about, so OT possessions are excluded from the fit
rather than modelled with a regulation-shaped clock.
"""
import json
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config
import rapm_lib as rl

REG_SECONDS = 2880.0        # 4 x 12:00
TRAIN, VAL = 0.8, 0.1       # remainder is test; split by game


def features(margin, t_rem, home_off):
    """Margin, time, and the interaction that makes a lead decisive as the clock
    runs out. sqrt(t) is the standard scaling: a 6-point lead with 2 minutes
    left is worth far more than the same lead in the first quarter."""
    t = np.maximum(t_rem, 1.0)
    return np.column_stack([
        margin,
        margin / np.sqrt(t),
        np.log1p(t),
        home_off,
    ])


def build_states(con):
    pl = pd.read_sql(
        "SELECT game_id, season, possession_number, home_off, pts "
        "FROM possession_lineups ORDER BY game_id, possession_number", con)
    poss = pd.read_sql(
        "SELECT game_id, possession_number, period, start_time FROM possessions",
        con)
    df = pl.merge(poss, on=["game_id", "possession_number"], how="inner")
    df = df.sort_values(["game_id", "possession_number"]).reset_index(drop=True)

    # running score BEFORE each possession
    df["home_pts"] = df.pts * df.home_off
    df["away_pts"] = df.pts * (1 - df.home_off)
    g = df.groupby("game_id", sort=False)
    df["home_before"] = g.home_pts.cumsum() - df.home_pts
    df["away_before"] = g.away_pts.cumsum() - df.away_pts
    df["margin"] = df.home_before - df.away_before

    # final margin per game -> outcome
    finals = g[["home_pts", "away_pts"]].sum()
    finals["home_win"] = (finals.home_pts > finals.away_pts).astype(int)
    ties = int((finals.home_pts == finals.away_pts).sum())
    df = df.merge(finals[["home_win"]], left_on="game_id", right_index=True)

    # seconds remaining in regulation (rapm_lib owns the clock arithmetic,
    # including OT period lengths, so it isn't re-derived here)
    elapsed = [rl.elapsed_tenths(int(p), c) / 10.0
               for p, c in zip(df.period, df.start_time)]
    df["t_rem"] = REG_SECONDS - np.asarray(elapsed, dtype=float)
    df = df[(df.period <= 4) & df.t_rem.notna() & (df.t_rem >= 0)]
    return df.reset_index(drop=True), ties


def main():
    con = sqlite3.connect(config.DB_PATH)
    df, ties = build_states(con)
    print(f"states: {len(df):,} regulation possessions, "
          f"{df.game_id.nunique():,} games ({ties} finished level in regulation)")

    games = np.array(sorted(df.game_id.unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(games)
    n = len(games)
    tr = set(games[: int(n * TRAIN)])
    va = set(games[int(n * TRAIN): int(n * (TRAIN + VAL))])
    split = np.where(df.game_id.isin(tr), "train",
                     np.where(df.game_id.isin(va), "val", "test"))

    X = features(df.margin.to_numpy(float), df.t_rem.to_numpy(float),
                 df.home_off.to_numpy(float))
    y = df.home_win.to_numpy(int)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X[split == "train"], y[split == "train"])
    df["wp"] = clf.predict_proba(X)[:, 1]

    def report(name):
        m = split == name
        p, a = df.wp.to_numpy()[m], y[m]
        brier = float(np.mean((p - a) ** 2))
        # calibration: does a predicted-70% state win ~70% of the time?
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        rows = []
        worst = 0.0
        for b in range(10):
            s = idx == b
            if s.sum() < 50:
                continue
            pred, act = float(p[s].mean()), float(a[s].mean())
            worst = max(worst, abs(pred - act))
            rows.append({"bin": round(bins[b] + 0.05, 2),
                         "pred": round(pred, 4), "actual": round(act, 4),
                         "n": int(s.sum())})
        print(f"  [{name}] n={int(m.sum()):,} brier={brier:.4f} "
              f"max|pred-actual|={worst:.4f}")
        return {"n": int(m.sum()), "brier": round(brier, 5),
                "maxCalErr": round(worst, 4), "bins": rows}

    print("fit + calibration:")
    rep = {k: report(k) for k in ("train", "val", "test")}

    # sanity: a tie with the ball, mid-game, should be near a coin flip; a
    # 20-point lead with a minute left should be ~certain.
    probe = features(np.array([0.0, 20.0, -20.0]),
                     np.array([1440.0, 60.0, 60.0]),
                     np.array([1.0, 1.0, 1.0]))
    pv = clf.predict_proba(probe)[:, 1]
    print(f"  probes: tie@24min={pv[0]:.3f}  +20@1min={pv[1]:.3f}  "
          f"-20@1min={pv[2]:.3f}")
    if not (0.4 < pv[0] < 0.6 and pv[1] > 0.97 and pv[2] < 0.03):
        raise ValueError(f"win-prob probes are implausible: {pv}")

    out = df[["game_id", "season", "possession_number", "period", "t_rem",
              "margin", "home_off", "home_win", "wp"]].copy()
    out["wp"] = out.wp.round(5)
    out.to_sql("winprob_states", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_wp_game "
                "ON winprob_states(game_id, possession_number)")

    meta = {
        "model": "logistic(margin, margin/sqrt(t), log1p(t), possession)",
        "spreadTerm": False,
        "spreadNote": ("no free licensed source of historical point spreads; "
                       "strength prior omitted, decisions are compared within "
                       "a single game state so the bias largely cancels"),
        "regulationOnly": True,
        "coef": dict(zip(["margin", "marginOverSqrtT", "logT", "possession"],
                         [round(float(c), 6) for c in clf.coef_[0]])),
        "intercept": round(float(clf.intercept_[0]), 6),
        "splits": rep,
        "tiedAfterRegulation": ties,
    }
    con.execute("DROP TABLE IF EXISTS winprob_meta")
    con.execute("CREATE TABLE winprob_meta (json TEXT)")
    con.execute("INSERT INTO winprob_meta VALUES (?)", (json.dumps(meta),))
    con.commit()
    con.close()
    print(f"winprob_states: {len(out):,} rows")


if __name__ == "__main__":
    main()
