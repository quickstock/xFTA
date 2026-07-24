"""Export the RAPM + lineup tables to the site JSON contract.

  site/public/rapm-NBA.json          meta, players (per season), pooled,
                                     lineups index (per season, top by |synergy|)
  site/public/lineups-NBA-{season}.json  per-lineup game log, on demand

Ids are stringified; teams via the shared TEAM_ABBREV map. This is a pure
read of the DB tables train_rapm / build_lineups wrote — no modeling here.
"""
import json
import sqlite3
from pathlib import Path

import pandas as pd

from config import DB_PATH

# stats.nba.com franchise id -> abbreviation (mirrors export_site_data's
# constant; kept local because that module runs an export at import time).
TEAM_ABBREV = {
    1610612737: "ATL", 1610612738: "BOS", 1610612739: "CLE", 1610612740: "NOP",
    1610612741: "CHI", 1610612742: "DAL", 1610612743: "DEN", 1610612744: "GSW",
    1610612745: "HOU", 1610612746: "LAC", 1610612747: "LAL", 1610612748: "MIA",
    1610612749: "MIL", 1610612750: "MIN", 1610612751: "BKN", 1610612752: "NYK",
    1610612753: "ORL", 1610612754: "IND", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612760: "OKC",
    1610612761: "TOR", 1610612762: "UTA", 1610612763: "MEM", 1610612764: "WAS",
    1610612765: "DET", 1610612766: "CHA",
}

OUT = Path(__file__).parent / "site" / "public"
QUALIFY_POSS = 2000      # poss_off + poss_def; rotation-player default floor
BOARD_MAX = 8000
BOARD_STEP = 250
LINEUPS_PER_SEASON = 60  # top by |synergy| for the board index
VERSION = 1
GENERATED = "2026-07-24"


def team_abbr(franchise_id) -> str:
    try:
        return TEAM_ABBREV.get(int(franchise_id), "UNK")
    except (TypeError, ValueError):
        return "UNK"


def player_rows(rapm: pd.DataFrame, season: str):
    d = rapm[rapm.season == season]
    out = []
    for r in d.itertuples():
        out.append({
            "id": str(int(r.player_id)), "name": r.player_name,
            "possOff": int(r.poss_off), "possDef": int(r.poss_def),
            "o": round(r.o, 2), "d": round(r.d, 2), "net": round(r.net, 2),
            "oP": round(r.o_p, 2), "dP": round(r.d_p, 2),
            "netP": round(r.net_p, 2),
            "seO": round(r.se_o, 2), "seD": round(r.se_d, 2),
        })
    out.sort(key=lambda x: x["netP"], reverse=True)
    return out


def main():
    con = sqlite3.connect(DB_PATH)
    rapm = pd.read_sql("SELECT * FROM rapm", con)
    lineup = pd.read_sql("SELECT * FROM lineup_season", con)
    lgames = pd.read_sql("SELECT * FROM lineup_games", con)
    rmeta = json.loads(pd.read_sql("SELECT json FROM rapm_meta", con).json.iloc[0])
    lmeta = json.loads(pd.read_sql("SELECT json FROM lineup_meta", con).json.iloc[0])
    names = dict(zip(rapm.player_id.astype(int), rapm.player_name))
    con.close()

    seasons = sorted(s for s in rapm.season.unique() if s != "pooled")

    def lineup_obj(r):
        pids = [p for p in r.lineup_id.split("-")]
        return {
            "lineupId": r.lineup_id,
            "playerIds": pids,
            "players": [names.get(int(p), p) for p in pids],
            "team": team_abbr(r.team),
            "poss": int(r.poss),
            "net100": round(r.net100, 2), "exp100": round(r.exp100, 2),
            "synergy100": round(r.synergy100, 2),
            "sqSynergy": (None if pd.isna(r.sq_synergy)
                          else round(r.sq_synergy, 4)),
        }

    lineups = {}
    for s in seasons:
        d = lineup[lineup.season == s].copy()
        d["absS"] = d.synergy100.abs()
        top = d.nlargest(LINEUPS_PER_SEASON, "absS")
        lineups[s] = [lineup_obj(r) for r in top.itertuples()]

    core = {
        "meta": {
            "layer": "lineups", "league": "NBA", "version": VERSION,
            "generated": GENERATED, "seasons": seasons,
            "qualifyPoss": QUALIFY_POSS, "boardMax": BOARD_MAX,
            "boardStep": BOARD_STEP,
            "lambda": rmeta["lambda"], "lambdaPooled": rmeta["lambdaPooled"],
            "cv": rmeta["cv"], "cvPooled": rmeta["cvPooled"],
            "collinear": rmeta["collinear"],
            "boxPriorR2": rmeta["boxPriorR2"],
            "synergyOOS": lmeta["oos"], "lineupFloorPoss": lmeta["floorPoss"],
            "skippedGames": sum(rmeta["skips"].values()),
            "totalGames": 7230, "gamesKept": rmeta["gamesKept"],
        },
        "players": {s: player_rows(rapm, s) for s in seasons},
        "pooled": player_rows(rapm, "pooled"),
        "lineups": lineups,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rapm-NBA.json").write_text(json.dumps(core, separators=(",", ":")))
    size = (OUT / "rapm-NBA.json").stat().st_size / 1e6
    print(f"wrote rapm-NBA.json ({size:.2f} MB): "
          f"{sum(len(v) for v in core['players'].values())} player-seasons, "
          f"{len(core['pooled'])} pooled, "
          f"{sum(len(v) for v in lineups.values())} lineup rows")

    # per-season lineup game logs (only lineups in the board index)
    for s in seasons:
        idx = {lu["lineupId"] for lu in lineups[s]}
        d = lgames[(lgames.season == s) & (lgames.lineup_id.isin(idx))]
        chunk = {}
        for lid, g in d.groupby("lineup_id"):
            chunk[lid] = {"games": [
                [r.game_id, int(r.poss_off), int(r.poss_def),
                 int(r.pts_for), int(r.pts_against)]
                for r in g.itertuples()]}
        (OUT / f"lineups-NBA-{s}.json").write_text(
            json.dumps(chunk, separators=(",", ":")))
    print(f"wrote {len(seasons)} lineup game-log chunks")


if __name__ == "__main__":
    main()
