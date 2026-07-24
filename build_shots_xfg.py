"""Persist per-shot out-of-fold xFG into shots_xfg.

The season cross-fit and anchoring are shot_value.py's, imported unchanged
(same model config, same features, same leak-free discipline) — this stage
only PERSISTS what that pipeline computes and discards, because the lineup
synergy decomposition needs look quality per shot, not per player-season.

    shots_xfg: game_id, event_id, xfg (anchored OOF P(make)),
               xpts (xfg x shot value, the FG-only look value)

Gate printed at the end: per-season sum(xpts) here vs the shot_value
table's exp_fg_pts totals (both sides derive from the same cross-fit, so
they must agree within float-and-row-order noise).
"""
import sqlite3

import numpy as np
import pandas as pd

from config import DB_PATH
from shot_value import cross_fit_xfg
from xfg_model import engineer


def load_with_event_id() -> pd.DataFrame:
    """shot_value.load_all() plus s.event_id (needed to key shots_xfg);
    same filters so the cross-fit sees the identical population."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """SELECT s.game_id, s.event_id, s.period,
                  s.seconds_remaining_in_period, s.shot_made,
                  s.shot_x, s.shot_y, s.shot_distance,
                  s.shot_zone_basic, s.shot_zone_area,
                  s.action_type, s.shot_type, s.score_margin, g.season
           FROM shots s JOIN games g ON g.GAME_ID = s.game_id
           WHERE s.shot_zone_basic IS NOT NULL AND s.shot_zone_basic != ''
             AND s.shot_x IS NOT NULL AND s.shot_y IS NOT NULL""",
        conn,
    )
    conn.close()
    return df


def main() -> None:
    df = engineer(load_with_event_id())
    print(f"loaded {len(df):,} FGA across {df['season'].nunique()} seasons")
    print("season cross-fit xFG% (same folds as shot_value.py):")
    df["xfg"] = cross_fit_xfg(df)

    # identical per-season anchoring to shot_value.py
    chk = df.groupby("season").agg(
        actual=("shot_made", "mean"), xfg=("xfg", "mean"))
    anchors = chk["actual"] / chk["xfg"]
    df["xfg"] = df["xfg"] * df["season"].map(anchors)

    pts = np.where(df["shot_type"].str.contains("3"), 3, 2)
    df["xpts"] = df["xfg"] * pts

    out = df[["game_id", "event_id", "xfg", "xpts"]].copy()
    out["xfg"] = out["xfg"].round(5)
    out["xpts"] = out["xpts"].round(5)

    conn = sqlite3.connect(DB_PATH)
    out.to_sql("shots_xfg", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_shots_xfg "
                 "ON shots_xfg(game_id, event_id)")
    conn.commit()
    print(f"wrote shots_xfg: {len(out):,} rows")

    # consistency gate vs the shot_value table (player-season aggregates)
    mine = (df.assign(season=df["season"])
              .groupby("season")["xpts"].sum())
    theirs = pd.read_sql(
        "SELECT season, SUM(exp_fg_pts) AS exp FROM shot_value "
        "GROUP BY season", conn).set_index("season")["exp"]
    conn.close()
    print("\nconsistency vs shot_value.exp_fg_pts (qualified players only,"
          " so theirs < mine; ratio must be stable across seasons):")
    cmp = pd.DataFrame({"shots_xfg_all": mine.round(0),
                        "shot_value_qualified": theirs.round(0)})
    cmp["ratio"] = (cmp.shot_value_qualified / cmp.shots_xfg_all).round(4)
    print(cmp.to_string())
    spread = cmp.ratio.max() - cmp.ratio.min()
    print(f"ratio spread across seasons: {spread:.4f} "
          f"({'OK' if spread < 0.03 else 'CHECK'})")


if __name__ == "__main__":
    main()
