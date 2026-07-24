"""leaguedashplayerstats (Per100Possessions) per season -> cache/box100/.

Feeds the RAPM box prior: 12 per-100 box columns per player-season. Six
requests total, cache-first; gentle pacing (this API family bans bursts).
"""
import os
import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

import config

OUT_DIR = os.path.join(config.CACHE_DIR, "box100")
COLS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN",
        "PTS", "FGA", "FG3A", "FTA", "OREB", "DREB", "AST", "TOV",
        "STL", "BLK", "PF"]


def pull_season(season: str) -> None:
    path = os.path.join(OUT_DIR, f"{season}.parquet")
    if os.path.exists(path):
        return
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, per_mode_detailed="Per100Possessions",
        season_type_all_star="Regular Season",
        timeout=config.NBA_API_TIMEOUT,
    ).get_data_frames()[0][COLS]
    df.to_parquet(path, index=False)
    print(f"  {season}: {len(df)} players")
    time.sleep(5.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for season in config.SEASONS:
        pull_season(season)
    print("box100 done")


if __name__ == "__main__":
    main()
