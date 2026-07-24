"""player_value -> site/public/value-NBA.json (Layer 2).

One core file, no per-season chunks: a player-season row is small and there is
no per-game series behind this layer.

`salaryAvailable` is per season and drives the front end. With no salary file
the export still ships the full production leaderboard (wins over replacement)
and the dollar fields stay null — never zero, never invented — so the UI can say
the surplus half is unavailable instead of rendering an empty money column.
"""
import datetime as dt
import json
import sqlite3
from pathlib import Path

import pandas as pd

import config
from export_rapm import TEAM_ABBREV

OUT = Path(__file__).parent / "site" / "public"
VERSION = 1
QUALIFY_POSS = 2000   # same rotation-player floor the lineups layer uses
BOARD_MAX = 8000
BOARD_STEP = 250


def main():
    con = sqlite3.connect(config.DB_PATH)
    df = pd.read_sql("SELECT * FROM player_value", con)
    meta_row = con.execute("SELECT json FROM value_meta").fetchone()
    vmeta = json.loads(meta_row[0])
    con.close()

    seasons = sorted(df.season.unique())
    players = {}
    for s in seasons:
        g = df[df.season == s].sort_values("war", ascending=False)
        rows = []
        for r in g.itertuples():
            teams = [TEAM_ABBREV.get(int(t), str(t))
                     for t in json.loads(r.teams) if int(t) > 0]
            rows.append({
                "id": str(int(r.player_id)),
                "name": r.player_name,
                "teams": teams,
                "poss": int(r.poss),
                "share": round(float(r.share), 4),
                "net": round(float(r.net_p), 2),
                "vorp": round(float(r.vorp), 2),
                "war": round(float(r.war), 2),
                "salary": None if pd.isna(r.salary) else float(r.salary),
                "value": None if pd.isna(r.value_usd) else float(r.value_usd),
                "surplus": (None if pd.isna(r.surplus_usd)
                            else float(r.surplus_usd)),
            })
        players[s] = rows

    core = {
        "meta": {
            "layer": "value", "league": "NBA", "version": VERSION,
            "generated": dt.datetime.now(dt.timezone.utc)
                           .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seasons": seasons,
            "qualifyPoss": QUALIFY_POSS, "boardMax": BOARD_MAX,
            "boardStep": BOARD_STEP,
            "replacementPer100": vmeta["replacementPer100"],
            "winsPerVorp": vmeta["winsPerVorp"],
            "impactMetric": vmeta["impactMetric"],
            "teamGames": vmeta["teamGames"],
            "salaryAvailable": vmeta["salaryAvailable"],
            "anySalary": any(vmeta["salaryAvailable"].values()),
            "costPerWin": vmeta["costPerWin"],
        },
        "players": players,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "value-NBA.json"
    path.write_text(json.dumps(core, separators=(",", ":")))
    print(f"wrote value-NBA.json ({path.stat().st_size / 1e6:.2f} MB): "
          f"{sum(len(v) for v in players.values())} player-seasons, "
          f"salary {'present' if core['meta']['anySalary'] else 'absent'}")


if __name__ == "__main__":
    main()
