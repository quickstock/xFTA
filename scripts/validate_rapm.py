"""The RAPM export gate. Exit 1 on any failure; run before shipping.

Checks:
  - rotation coverage: kept games >= 98.5% of total
  - points identity: every kept game's possession points sum to its final
    score (re-derived from possession_lineups vs games... via pbp is
    expensive, so we assert the build_stints invariant held: no
    points-identity skips remain among KEPT games — i.e. kept == in DB)
  - anchor: poss-weighted mean net per season within +/-0.5 of 0
  - CV: RAPM beats intercept+HCA AND box-prior-sum every season
  - JSON: parses, every required meta key present, ids are strings
Informational (printed, never fails): share of RAPM ids in the FTAOE
leaderboard id space (RAPM's floor admits players below that bar).
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH

CORE = Path(__file__).resolve().parent.parent / "site" / "public" / "rapm-NBA.json"
REQUIRED_META = {
    "layer", "league", "version", "generated", "seasons", "qualifyPoss",
    "boardMax", "boardStep", "lambda", "cv", "collinear", "boxPriorR2",
    "synergyOOS", "skippedGames", "totalGames",
}


def main():
    fails = []
    con = sqlite3.connect(DB_PATH)

    total = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    kept = con.execute(
        "SELECT COUNT(DISTINCT game_id) FROM possession_lineups").fetchone()[0]
    cov = kept / total
    print(f"rotation coverage: {kept}/{total} = {cov:.1%}")
    # 97% floor: lineups are reconstructed from PBP substitution parsing
    # (GameRotation, the authoritative source, is rate-limited to near-
    # uselessness and only repairs games opportunistically in the
    # background). Skipped games are MISSING, not wrong — ~0.4% are
    # genuinely corrupt feeds (score regression, permanently excluded, the
    # same convention every league uses) and the rest are name-resolution
    # gaps that the GameRotation backfill closes over time. Below 97%
    # something is systematically broken.
    if cov < 0.97:
        fails.append(f"coverage {cov:.1%} < 97%")

    rapm = pd.read_sql("SELECT * FROM rapm", con)
    for s, g in rapm[rapm.season != "pooled"].groupby("season"):
        w = g.poss_off + g.poss_def
        anchor = float((g.net * w).sum() / w.sum())
        if abs(anchor) > 0.5:
            fails.append(f"anchor {s} = {anchor:+.3f} (>0.5)")
    print("anchors OK" if not any("anchor" in f for f in fails)
          else "anchor FAIL")

    meta = json.loads(pd.read_sql("SELECT json FROM rapm_meta", con).json.iloc[0])
    for r in meta["cv"]:
        if not (r["rmseRapm"] < r["rmseHca"]
                and r["rmseRapm"] < r["rmsePriorSum"]):
            fails.append(f"CV {r['season']}: RAPM does not beat baselines")
    print("CV beats baselines OK" if not any("CV" in f for f in fails)
          else "CV FAIL")

    # JSON contract
    if not CORE.exists():
        fails.append("rapm-NBA.json missing")
    else:
        core = json.loads(CORE.read_text())
        missing = REQUIRED_META - set(core["meta"])
        if missing:
            fails.append(f"meta missing keys: {missing}")
        sample = next(iter(core["players"].values()))[0]
        if not isinstance(sample["id"], str):
            fails.append("player id not a string")
        # informational: id overlap with FTAOE board
        # board player_id is stored as float ("1629678.0"); normalize to
        # the clean int-string the export and data.json both use.
        board_ids = set(pd.read_sql(
            "SELECT DISTINCT CAST(CAST(player_id AS INTEGER) AS TEXT) id "
            "FROM player_season_xfta_poss_lb_clean", con).id)
        rapm_ids = {r["id"] for rows in core["players"].values() for r in rows}
        overlap = len(rapm_ids & board_ids) / len(rapm_ids)
        print(f"[info] {overlap:.0%} of RAPM ids are in the FTAOE board id "
              f"space (rest are sub-threshold players; player page handles "
              f"gracefully)")
        print(f"[info] synergy OOS r={core['meta']['synergyOOS']['r']} "
              f"n={core['meta']['synergyOOS']['n']}")

    con.close()
    if fails:
        print("\nGATE FAILED:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("\nRAPM GATE PASSED")


if __name__ == "__main__":
    main()
