"""team_defense -> site/public/defense-NBA.json (Layer 4, team level).

Deliberately a team artifact. The player-level metric was built and rejected on
evidence (WO-6): 0.348 split-half reliability and a leaderboard with non-
defenders at the top, because attributing every shot equally to five defenders
is a team measure wearing a player's name. D-RAPM in Layer 1 is the
regression-adjusted player signal and already ships with intervals and tiers.

The pillars are ordered by measured reliability, not by which sounds best, and
each row carries its pillar's Spearman-Brown so the front end cannot present the
noisy one as though it were the solid one:

  quality forced        0.981   how bad the shots a defence concedes are
  rim deterrence        0.964   restricted-area attempts faced vs league
  conversion suppression 0.717  opponents shooting below expectation

`privileged` marks this NBA-only with no European equivalent, so the UI badges
it rather than implying the other eight leagues simply have not been processed.
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

# every active league except the NBA: none has the shot-level data this needs
LEAGUES_UNAVAILABLE = ["EL", "EUC", "ACB", "LBA", "BBL", "GBL", "ABA", "WNBA"]


def main():
    con = sqlite3.connect(config.DB_PATH)
    df = pd.read_sql("SELECT * FROM team_defense", con)
    tmeta = json.loads(
        con.execute("SELECT json FROM team_defense_meta").fetchone()[0])
    con.close()

    seasons = sorted(df.season.unique())
    teams = {}
    for s in seasons:
        g = df[df.season == s].sort_values("quality_forced", ascending=False)
        teams[s] = [{
            "team": TEAM_ABBREV.get(int(r.def_team), str(int(r.def_team))),
            "teamId": str(int(r.def_team)),
            "defPoss": int(r.def_poss),
            "shotsFaced": int(r.shots_faced),
            # the reliable pillars first, mirroring how the board reads
            "qualityForced": round(float(r.quality_forced), 4),
            "qualityConceded": round(float(r.quality_conceded), 4),
            "deterrence": round(float(r.deterrence) * 100, 2),
            "rimRate": round(float(r.rim_rate) * 100, 2),
            "suppression": round(float(r.suppression), 2),
            "ptsAllowed100": round(float(r.pts_allowed_100), 2),
        } for r in g.itertuples()]

    rel = tmeta["reliability"]
    core = {
        "meta": {
            "layer": "defense", "league": "NBA", "version": VERSION,
            "generated": dt.datetime.now(dt.timezone.utc)
                           .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "privileged": True,
            "leaguesUnavailable": LEAGUES_UNAVAILABLE,
            "unavailableReason": (
                "This layer needs shot-level expected-points against a "
                "reconstructed five-man defensive lineup. No European league "
                "publishes the possession-level data it requires, so there is "
                "nothing to approximate from — this is absent, not pending."),
            "aggregationLevel": "team-season",
            "seasons": seasons,
            "nTeamSeasons": tmeta["nTeamSeasons"],
            # ordered by measured reliability; the UI renders in this order
            "pillars": [
                {"key": "qualityForced",
                 "label": "Quality forced",
                 "unit": "expected points per shot faced, vs league mean",
                 "spearmanBrown": rel["quality"]["spearmanBrown"],
                 "note": ("how bad the shots this defence concedes are. The "
                          "most reliable of the three and the part a defence "
                          "genuinely controls.")},
                {"key": "deterrence",
                 "label": "Rim deterrence",
                 "unit": "percentage points of shots faced at the rim, vs league",
                 "spearmanBrown": rel["rim"]["spearmanBrown"],
                 "note": ("how well the defence keeps opponents away from the "
                          "restricted area.")},
                {"key": "suppression",
                 "label": "Conversion suppression",
                 "unit": "points below expectation per 100 defensive possessions",
                 "spearmanBrown": rel["supp"]["spearmanBrown"],
                 "note": ("whether opponents shot worse than their shots were "
                          "worth. Measurably the noisiest pillar at 0.72, so a "
                          "single season of it is closer to a hint than a "
                          "rating.")},
            ],
            "reliability": rel,
            "playerLevelRejected": tmeta["playerLevelRejected"],
            "notModelled": tmeta["notModelled"],
            "leagueCentred": tmeta["leagueCentred"],
        },
        "teams": teams,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "defense-NBA.json"
    path.write_text(json.dumps(core, separators=(",", ":")))
    print(f"wrote defense-NBA.json ({path.stat().st_size / 1e3:.1f} KB): "
          f"{sum(len(v) for v in teams.values())} team-seasons across "
          f"{len(seasons)} seasons")


if __name__ == "__main__":
    main()
