"""Layer 4, team level — the part of WO-6 the validation actually supports.

The player-level version of this metric was built, tested, and rejected: shots
attributed equally to five defenders makes it a team measure wearing a player's
name, split-half reliability was 0.35, and its leaderboard put Ja Morant among
the league's best defenders. See WO-6 in the calibration PROGRESS log.

Five-way credit sharing is not a confound when the unit *is* the team, so the
same substrate is defensible here. But "defensible in principle" is not
evidence, so this script tests the team version the same way the player version
was tested and reports the result rather than assuming it improved.

It decomposes team defence into two parts that answer different questions:

  **quality conceded** — expected points per shot faced, from the xPTS model.
  Low means opponents were forced into bad shots. This is shot *selection*
  forced, and it is the part a defence controls.

  **conversion suppression** — (expected − actual) points per 100 defensive
  possessions, positive meaning opponents shot worse than their shots were
  worth. Whether this is skill or opponent variance is an empirical question,
  so reliability is measured for both and the export leads with whichever
  earns it.

Not modelled, named rather than approximated: turnovers forced and defensive
rebounding (they live in the play-by-play parquet, not SQL), and the
alteration-vs-avoidance distinction, which no public data supports since the
NBA withdrew per-shot closest-defender data.
"""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from build_defense import RIM_ZONE, shot_possessions

MIN_SHOTS_HALF = 500      # per half, for the split-half reliability test


def defending_team(joined, con):
    """Attach the defending team to every attributed shot.

    `possession_lineups` records who had the ball, not who was defending, but a
    game has exactly two teams, so the defender is the other one. Derived rather
    than assumed: any game whose possessions name more or fewer than two teams
    is dropped and counted, so a bad assumption would show up as a skip count
    instead of silently mislabelling a team's defence.
    """
    poss = pd.read_sql(
        "SELECT game_id, possession_number, off_team FROM possession_lineups",
        con)
    teams = poss.groupby("game_id").off_team.unique()
    pair = {g: list(t) for g, t in teams.items() if len(t) == 2}
    dropped = int(len(teams) - len(pair))

    j = joined.merge(poss, on=["game_id", "possession_number"], how="inner")
    j = j[j.game_id.isin(pair)]
    j["def_team"] = [
        pair[g][1] if pair[g][0] == o else pair[g][0]
        for g, o in zip(j.game_id, j.off_team)
    ]
    return j, dropped


def spearman_brown(r):
    return (2 * r) / (1 + r) if r > -1 else float("nan")


def split_half(j):
    """Odd-vs-even shots faced, per team-season, for both pillars.

    The identical test that failed the player metric at 0.348. Shots are split
    by parity of their within-team-season order, which is arbitrary with respect
    to opponent and date, so the two halves face comparable schedules.
    """
    j = j.sort_values(["season", "def_team", "game_id", "event_id"]).copy()
    j["k"] = j.groupby(["season", "def_team"]).cumcount() % 2
    g = j.groupby(["season", "def_team", "k"]).agg(
        n=("actual", "size"),
        allowed=("actual", "sum"),
        expected=("expected", "sum"),
        rim=("zone", lambda z: float((z == RIM_ZONE).mean())),
    ).reset_index()
    g["quality"] = g.expected / g.n
    g["supp"] = (g.expected - g.allowed) / g.n

    wide = g.pivot_table(index=["season", "def_team"], columns="k",
                         values=["n", "quality", "supp", "rim"])
    wide = wide[(wide[("n", 0)] >= MIN_SHOTS_HALF)
                & (wide[("n", 1)] >= MIN_SHOTS_HALF)]
    out = {"nTeamSeasons": int(len(wide)), "minShotsPerHalf": MIN_SHOTS_HALF}
    for pillar in ("quality", "supp", "rim"):
        r = float(np.corrcoef(wide[(pillar, 0)], wide[(pillar, 1)])[0, 1])
        out[pillar] = {"halfR": round(r, 4),
                       "spearmanBrown": round(spearman_brown(r), 4)}
    return out


def main():
    con = sqlite3.connect(config.DB_PATH)
    joined, unmatched, n_shots = shot_possessions(con)
    j, dropped_games = defending_team(joined, con)
    print(f"shots attributed: {len(j):,} of {n_shots:,} "
          f"({unmatched:,} unmatched to a possession, "
          f"{dropped_games} games dropped for not naming exactly two teams)")

    rel = split_half(j)
    print("\n=== split-half reliability, team-season "
          f"(n={rel['nTeamSeasons']}) ===")
    for pillar, label in (("quality", "quality conceded"),
                          ("supp", "conversion suppression"),
                          ("rim", "rim rate faced")):
        print(f"  {label:24s} r={rel[pillar]['halfR']:+.3f}  "
              f"Spearman-Brown={rel[pillar]['spearmanBrown']:.3f}")
    print("  (the player-level metric this replaces scored 0.348)")

    # defensive possessions per team-season
    poss = pd.read_sql(
        "SELECT game_id, possession_number, season, off_team "
        "FROM possession_lineups", con)
    teams = poss.groupby("game_id").off_team.unique()
    pair = {g: list(t) for g, t in teams.items() if len(t) == 2}
    poss = poss[poss.game_id.isin(pair)].copy()
    poss["def_team"] = [
        pair[g][1] if pair[g][0] == o else pair[g][0]
        for g, o in zip(poss.game_id, poss.off_team)
    ]
    dposs = poss.groupby(["season", "def_team"]).size().rename("def_poss") \
                .reset_index()

    agg = j.groupby(["season", "def_team"]).agg(
        shots_faced=("actual", "size"),
        pts_allowed=("actual", "sum"),
        pts_expected=("expected", "sum"),
        rim_faced=("zone", lambda z: int((z == RIM_ZONE).sum())),
    ).reset_index().merge(dposs, on=["season", "def_team"], how="inner")

    agg["quality_conceded"] = agg.pts_expected / agg.shots_faced
    agg["suppression"] = ((agg.pts_expected - agg.pts_allowed)
                          / agg.def_poss * 100)
    agg["rim_rate"] = agg.rim_faced / agg.shots_faced
    agg["pts_allowed_100"] = agg.pts_allowed / agg.def_poss * 100

    # league mean per season, so a season's pace or rule environment does not
    # leak into a team's number
    for col in ("quality_conceded", "rim_rate"):
        m = agg.groupby("season")[col].transform("mean")
        agg[f"{col}_vs_lg"] = agg[col] - m
    # signed so positive is good defence in both cases
    agg["quality_forced"] = -agg.quality_conceded_vs_lg
    agg["deterrence"] = -agg.rim_rate_vs_lg

    keep = ["season", "def_team", "def_poss", "shots_faced", "rim_faced",
            "pts_allowed", "pts_expected", "quality_conceded",
            "quality_forced", "suppression", "rim_rate", "deterrence",
            "pts_allowed_100"]
    agg[keep].to_sql("team_defense", con, if_exists="replace", index=False)

    meta = {
        "unit": "quality: expected points per shot faced; suppression: points "
                "below expectation per 100 defensive possessions",
        "reliability": rel,
        "playerLevelRejected": {
            "spearmanBrown": 0.348,
            "why": ("shots split five ways across defenders makes the player "
                    "version a team metric with shared credit; its leaderboard "
                    "placed non-defenders near the top. D-RAPM in Layer 1 is "
                    "the regression-adjusted player defensive signal."),
        },
        "notModelled": [
            "turnovers forced and defensive rebounding — they live in the "
            "play-by-play parquet rather than SQL",
            "shot alteration vs shot avoidance: a defence that forces misses "
            "and one that prevents attempts look identical here, and no public "
            "data separates them since per-shot closest-defender data was "
            "withdrawn",
        ],
        "leagueCentred": ("quality conceded and rim rate are expressed against "
                          "the same season's league mean, so pace and rule "
                          "changes do not leak into a team's figure"),
        "nTeamSeasons": int(len(agg)),
        "seasons": sorted(agg.season.unique().tolist()),
    }
    con.execute("DROP TABLE IF EXISTS team_defense_meta")
    con.execute("CREATE TABLE team_defense_meta (json TEXT)")
    con.execute("INSERT INTO team_defense_meta VALUES (?)", (json.dumps(meta),))
    con.commit()

    latest = agg[agg.season == meta["seasons"][-1]]
    print(f"\n=== {meta['seasons'][-1]} best quality forced (top 6) ===")
    from export_rapm import TEAM_ABBREV
    for r in latest.nlargest(6, "quality_forced").itertuples():
        print(f"  {TEAM_ABBREV.get(int(r.def_team), r.def_team)}  "
              f"quality forced {r.quality_forced:+.4f} pts/shot vs league, "
              f"suppression {r.suppression:+.2f}/100")
    con.close()
    print(f"\nteam_defense: {len(agg)} team-seasons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
