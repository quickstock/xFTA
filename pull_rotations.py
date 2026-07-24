"""GameRotation -> cache/rotations/{gid}.parquet. Cache-first, resumable.

Failures are logged and skipped (re-run to retry), same convention as
pull.py. Rotation rows are the authoritative on/off record: IN/OUT_TIME_REAL
are tenths of a second of elapsed game time (Q1 start = 0, regulation end
= 28800, each OT adds 3000).
"""
import os
import sqlite3
import time

import pandas as pd
from nba_api.stats.endpoints import gamerotation

import config

OUT_DIR = os.path.join(config.CACHE_DIR, "rotations")


def pull_one(game_id: str) -> None:
    path = os.path.join(OUT_DIR, f"{game_id}.parquet")
    if os.path.exists(path):
        return
    r = gamerotation.GameRotation(game_id=game_id, timeout=config.NBA_API_TIMEOUT)
    away, home = r.get_data_frames()[0], r.get_data_frames()[1]
    df = pd.concat([away, home], ignore_index=True)[
        ["TEAM_ID", "PERSON_ID", "PLAYER_FIRST", "PLAYER_LAST",
         "IN_TIME_REAL", "OUT_TIME_REAL"]
    ]
    df.to_parquet(path, index=False)
    time.sleep(config.NBA_API_SLEEP)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    game_ids = [r[0] for r in con.execute(
        "SELECT GAME_ID FROM games ORDER BY GAME_ID")]
    con.close()
    fails = []
    for i, gid in enumerate(game_ids, 1):
        for attempt in range(config.NBA_API_RETRIES):
            try:
                pull_one(gid)
                break
            except Exception as e:  # log-and-continue: re-run to retry
                if attempt == config.NBA_API_RETRIES - 1:
                    fails.append((gid, repr(e)))
                else:
                    time.sleep(2 * (attempt + 1))
        if i % 200 == 0:
            print(f"  {i}/{len(game_ids)}", flush=True)
    print(f"done, {len(fails)} failures")
    for gid, err in fails[:10]:
        print(f"  FAIL {gid}: {err}")


if __name__ == "__main__":
    main()
