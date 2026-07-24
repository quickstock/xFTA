"""Layer 4 (WO-6) — defensive suppression and rim deterrence. PRIVILEGED, NBA-ONLY.

Every shot is attributed to the five defenders who were on the floor for it, using
the possession windows `rapm_lib` already owns, and scored against what the xPTS
model expected that shot to be worth. A defender who faces shots worth 1.10 expected
points and allows 1.05 is suppressing; one who allows 1.15 is not.

Two pillars, following the TDPM / Franks framing:

  **suppression** — points allowed below expectation per 100 defensive possessions,
  signed so positive is good defence. This is the part public data supports well.

  **rim deterrence** — the rate at which opponents attempt shots in the restricted
  area while a player is on the floor, against the league rate, signed so positive
  means fewer rim attempts faced.

**What this cannot do, stated in the artifact and not only here:** it cannot separate
shot *alteration* from shot *avoidance*. A rim protector who makes attackers miss and
one who makes them stop coming both look good, and no public data distinguishes them —
the per-shot closest-defender field that would has never been public. Turnover-forcing
and defensive rebounding are also absent: they live in the play-by-play parquet rather
than in SQL, so including them would mean a second ingestion pass. They are named as
missing rather than approximated.

Both metrics get empirical-Bayes shrinkage toward the league mean by possessions, so
a 200-possession sample does not outrank a 3,000-possession one on noise.

NBA only. There is no European equivalent and none is approximated; the export marks
this privileged so the UI can badge it.
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

RIM_ZONE = "Restricted Area"
# EB shrinkage constant, in defensive possessions: the sample size at which a
# player's own rate gets equal weight with the league mean.
EB_POSS = 1500.0
MIN_POSS_REPORT = 500


def shot_possessions(con):
    """Attribute every shot to a possession, then to the five defenders."""
    shots = pd.read_sql(
        """SELECT s.game_id, s.event_id, s.period,
                  s.seconds_remaining_in_period AS secs,
                  s.shot_zone_basic AS zone,
                  s.shot_made * (CASE WHEN s.shot_type LIKE '%3PT%'
                                      THEN 3 ELSE 2 END) AS actual,
                  x.xpts AS expected
           FROM shots s
           JOIN shots_xfg x ON x.game_id = s.game_id AND x.event_id = s.event_id
           WHERE s.shot_zone_basic IS NOT NULL""", con)
    poss = pd.read_sql(
        "SELECT game_id, period, possession_number, start_time, end_time "
        "FROM possessions", con)
    lineups = pd.read_sql(
        "SELECT game_id, season, possession_number, d1, d2, d3, d4, d5 "
        "FROM possession_lineups", con)

    out = []
    unmatched = 0
    for gid, sg in shots.groupby("game_id", sort=False):
        pg = poss[poss.game_id == gid]
        if pg.empty:
            continue
        windows = rl.possession_windows(pg)
        # shot elapsed tenths, from period + clock remaining
        el = np.array([rl.elapsed_tenths(int(p), _clock(sec))
                       for p, sec in zip(sg.period, sg.secs)], dtype=float)
        pn = rl.assign_possession_elapsed(el, windows)
        sub = sg.assign(possession_number=pn)
        unmatched += int((pn < 0).sum())
        out.append(sub[sub.possession_number >= 0])
    joined = pd.concat(out, ignore_index=True).merge(
        lineups, on=["game_id", "possession_number"], how="inner")
    return joined, unmatched, len(shots)


def _clock(secs_remaining):
    """seconds_remaining_in_period -> the ISO clock string rapm_lib parses."""
    s = float(secs_remaining or 0)
    return f"PT{int(s // 60):02d}M{s % 60:05.2f}S"


def defender_long(joined):
    """One row per (shot, defender): five rows per shot."""
    frames = []
    for slot in ("d1", "d2", "d3", "d4", "d5"):
        frames.append(joined[["season", slot, "actual", "expected", "zone"]]
                      .rename(columns={slot: "player_id"}))
    return pd.concat(frames, ignore_index=True)


def main():
    con = sqlite3.connect(config.DB_PATH)
    joined, unmatched, n_shots = shot_possessions(con)
    print(f"shots attributed to a possession: {len(joined):,} of {n_shots:,} "
          f"({unmatched:,} unmatched)")

    long = defender_long(joined)
    agg = long.groupby(["season", "player_id"]).agg(
        shots_faced=("actual", "size"),
        pts_allowed=("actual", "sum"),
        pts_expected=("expected", "sum"),
        rim_faced=("zone", lambda z: int((z == RIM_ZONE).sum())),
    ).reset_index()

    # defensive possessions on the floor
    lineups = pd.read_sql(
        "SELECT season, d1, d2, d3, d4, d5 FROM possession_lineups", con)
    dp = pd.concat([lineups[["season", s]].rename(columns={s: "player_id"})
                    for s in ("d1", "d2", "d3", "d4", "d5")],
                   ignore_index=True)
    dposs = dp.groupby(["season", "player_id"]).size().rename("def_poss") \
              .reset_index()
    d = agg.merge(dposs, on=["season", "player_id"], how="inner")
    d = d[d.player_id > 0]

    # raw rates. suppression signed so POSITIVE = allowed fewer points than
    # expected = good defence.
    d["supp_raw"] = -(d.pts_allowed - d.pts_expected) / d.def_poss * 100
    d["rim_rate"] = d.rim_faced / d.def_poss

    # empirical-Bayes shrinkage toward the season's league mean
    rows = []
    for season, g in d.groupby("season"):
        lg_supp = float(np.average(g.supp_raw, weights=g.def_poss))
        lg_rim = float(np.average(g.rim_rate, weights=g.def_poss))
        w = g.def_poss / (g.def_poss + EB_POSS)
        g = g.assign(
            suppression=lg_supp + w * (g.supp_raw - lg_supp),
            # positive = fewer rim attempts faced than league rate, per 100
            deterrence=-(lg_rim + w * (g.rim_rate - lg_rim) - lg_rim) * 100,
            leagueSupp=lg_supp, leagueRimRate=lg_rim, ebWeight=w)
        rows.append(g)
    d = pd.concat(rows, ignore_index=True)

    # ---- validation: split-half stability within season
    print("\n=== validation: split-half stability (odd vs even shots faced) ===")
    long2 = long.copy()
    long2["half"] = np.arange(len(long2)) % 2
    h = long2.groupby(["season", "player_id", "half"]).agg(
        a=("actual", "sum"), e=("expected", "sum"), n=("actual", "size")
    ).reset_index()
    h["rate"] = -(h.a - h.e) / h.n
    w = h.pivot_table(index=["season", "player_id"], columns="half",
                      values="rate")
    cnt = h.pivot_table(index=["season", "player_id"], columns="half",
                        values="n")
    keep = (cnt.min(axis=1) >= 150) & w.notna().all(axis=1)
    r = float(np.corrcoef(w[keep][0], w[keep][1])[0, 1])
    sb = 2 * r / (1 + r)
    print(f"  n={int(keep.sum())} player-seasons with >=150 shots faced per half")
    print(f"  half-half r = {r:+.4f}  ->  Spearman-Brown {sb:+.4f}")
    print("  interpretation: this is how repeatable the suppression signal is "
          "within a season.\n  Low values mean the metric is mostly noise at "
          "these sample sizes — reported either way.")

    qual = d[d.def_poss >= MIN_POSS_REPORT]
    print(f"\n=== top 12 suppression, 2024-25 (>= {MIN_POSS_REPORT} def poss) ===")
    names = pd.read_sql(
        "SELECT DISTINCT player_id, player_name FROM player_season", con)
    top = qual[qual.season == "2024-25"].merge(names, on="player_id", how="left") \
              .sort_values("suppression", ascending=False).head(12)
    for r_ in top.itertuples():
        print(f"  {str(r_.player_name)[:24]:<25} supp {r_.suppression:+6.2f}"
              f"  deter {r_.deterrence:+6.2f}  def poss {int(r_.def_poss):>5}"
              f"  shots faced {int(r_.shots_faced):>5}")

    keep_cols = ["season", "player_id", "def_poss", "shots_faced", "rim_faced",
                 "pts_allowed", "pts_expected", "supp_raw", "suppression",
                 "rim_rate", "deterrence", "ebWeight"]
    d[keep_cols].to_sql("defense_nba", con, if_exists="replace", index=False)

    meta = {
        "privileged": True,
        "league": "NBA",
        "leaguesUnavailable": ["EL", "EUC", "ACB", "LBA", "BBL", "GBL", "ABA",
                               "WNBA"],
        "unavailableReason": ("no European or WNBA equivalent is built and none is "
                             "approximated; per-shot defender data has never been "
                             "public and the aggregate dashboards are NBA-only"),
        "ebPossConstant": EB_POSS,
        "minPossReport": MIN_POSS_REPORT,
        "splitHalfR": round(r, 4),
        "splitHalfSpearmanBrown": round(sb, 4),
        "shotsAttributed": int(len(joined)),
        "shotsUnmatched": int(unmatched),
        "cannotSeparate": ("shot alteration from shot avoidance — a rim protector "
                           "who forces misses and one who stops attempts from "
                           "happening look alike here"),
        "notIncluded": ["turnover forcing", "defensive rebounding"],
        "notIncludedReason": ("both live in the play-by-play parquet rather than "
                              "SQL; naming them as absent beats approximating "
                              "them"),
    }
    con.execute("DROP TABLE IF EXISTS defense_meta")
    con.execute("CREATE TABLE defense_meta (json TEXT)")
    con.execute("INSERT INTO defense_meta VALUES (?)", (json.dumps(meta),))
    con.commit()
    con.close()
    print(f"\ndefense_nba: {len(d):,} player-seasons "
          f"({len(qual):,} above the reporting floor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
