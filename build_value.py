"""Layer 2 — player value and contract surplus. rapm + possession_lineups
-> player_value, value_meta.

Template is Basketball-Reference's VORP/BPM with one deliberate substitution:
**RAPM stands in for BPM.** Two reasons, in order of importance.

  1. BPM is a box-score *estimate* of a player's per-100 net impact. RAPM
     measures the same quantity by regression on who was actually on the floor,
     which is the stronger instrument and one this project already fits
     (Layer 1). Using BPM here would mean importing a weaker proxy for a
     number we own.
  2. BPM would have to be scraped from Basketball-Reference, and
     `tests/test_no_br_scaling.py` fails the build on any basketball-reference
     reference anywhere in the tree — a guard added after BR season totals
     corrupted the FTA target. That guard is not worth weakening for
     convenience.

Everything else follows the published formulas:

    VORP = (impact - replacement) x possession_share x (team_games / 82)
    wins_over_replacement = VORP x 2.70

with replacement fixed at -2.0 per 100, the BPM convention. RAPM is anchored so
the league sits at 0, the same centering BPM uses, so the replacement level
transfers directly. `possession_share` is a player's share of his team's
possessions, summed across teams so a midseason trade neither double-counts nor
drops him. The (team_games / 82) factor prorates the short 2020-21 season.

**Salaries are optional and absent by default.** Surplus needs a per-player
salary, which this project has no licensed free source for. Drop a CSV at
`cache/salaries/{season}.csv` with columns `player_id,salary` and every surplus
figure computes on the next run; without it the export sets
`salaryAvailable: false` and the front end says so rather than showing an empty
or invented dollar column — the same honest mode the spec requires for
EuroLeague, which publishes no per-player salaries at all.
"""
import json
import os
import sqlite3

import pandas as pd

import config

REPLACEMENT_PER_100 = -2.0   # BPM's replacement level, same per-100 net scale
WINS_PER_VORP = 2.70         # Basketball-Reference's VORP -> wins constant
FULL_SEASON_GAMES = 82
SALARY_DIR = os.path.join(config.CACHE_DIR, "salaries")


def load_salaries(season):
    """{player_id: salary} for a season, or {} when no file is present."""
    path = os.path.join(SALARY_DIR, f"{season}.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "player_id" not in cols or "salary" not in cols:
        raise ValueError(
            f"{path}: need columns player_id,salary (got {list(df.columns)})")
    df = df[[cols["player_id"], cols["salary"]]]
    df.columns = ["player_id", "salary"]
    df = df.dropna()
    return {int(r.player_id): float(r.salary) for r in df.itertuples()}


def possession_shares(pl):
    """Per (season, player_id): possessions, and share of his teams' floor time.

    A player is credited once per possession he is on the floor for, on either
    end, so the team denominator has to count both ends too — its possessions as
    offense plus as defence. (Dividing both-ends player counts by offence-only
    team counts double-counts and pushes shares past 1.) Shares are summed per
    team, which handles a midseason trade without double counting or dropping.
    """
    team_off = (pl.groupby(["season", "off_team"]).size()
                .rename("n").reset_index()
                .rename(columns={"off_team": "team"}))
    team_def = (pl.groupby(["season", "def_team"]).size()
                .rename("n").reset_index()
                .rename(columns={"def_team": "team"}))
    team_poss = (pd.concat([team_off, team_def], ignore_index=True)
                 .groupby(["season", "team"]).n.sum()
                 .rename("team_poss").reset_index())

    frames = []
    for side, team_col in (("o", "off_team"), ("d", "def_team")):
        slots = [f"{side}{i}" for i in range(1, 6)]
        for slot in slots:
            frames.append(pl[["season", team_col, slot]].rename(
                columns={team_col: "team", slot: "player_id"}))
    long = pd.concat(frames, ignore_index=True)
    pp = (long.groupby(["season", "team", "player_id"]).size()
          .rename("poss").reset_index())
    pp = pp.merge(team_poss, on=["season", "team"], how="left")
    pp["share"] = pp.poss / pp.team_poss

    out = (pp.groupby(["season", "player_id"])
           .agg(poss=("poss", "sum"), share=("share", "sum"),
                teams=("team", lambda s: sorted(set(int(x) for x in s))))
           .reset_index())
    return out


def main():
    con = sqlite3.connect(config.DB_PATH)
    # The defending team is whichever of the game's two teams isn't on offence.
    # Derived on ONE frame: joining a separately-queried copy and assigning the
    # result positionally silently misaligns if the merge duplicates a row or
    # the driver returns a different order.
    pl = pd.read_sql(
        "SELECT game_id, season, off_team, o1,o2,o3,o4,o5, d1,d2,d3,d4,d5 "
        "FROM possession_lineups", con)
    teams_per_game = pd.read_sql(
        "SELECT GAME_ID game_id, home_team_id, away_team_id FROM games", con)
    if teams_per_game.game_id.duplicated().any():
        raise ValueError("games.GAME_ID is not unique; the join below would "
                         "duplicate possessions")
    n_before = len(pl)
    pl = pl.merge(teams_per_game, on="game_id", how="left")
    if len(pl) != n_before:
        raise ValueError(f"possession join changed row count "
                         f"{n_before} -> {len(pl)}")
    pl["def_team"] = pl.away_team_id.where(
        pl.off_team == pl.home_team_id, pl.home_team_id)

    # Team id 0 is the pipeline's long-standing "couldn't attribute" marker (the
    # FTAOE export skips it too). Left in, it becomes a tiny bogus team whose
    # denominator makes a player's summed share exceed 1. These possessions have
    # no team to be a share OF, so they leave the share basis entirely.
    unattributed = ((pl.off_team <= 0) | (pl.def_team <= 0)
                    | pl.off_team.isna() | pl.def_team.isna())
    if unattributed.any():
        print(f"dropping {int(unattributed.sum())} of {len(pl)} possessions "
              f"with an unattributable team from the share basis")
        pl = pl[~unattributed]

    shares = possession_shares(pl)
    # A player cannot be on the floor for more than all of his team's
    # possessions; anything above 1 means the team denominator or the
    # offence/defence attribution is wrong.
    worst = shares.share.max()
    if worst > 1.001:
        bad = shares[shares.share > 1.001]
        raise ValueError(f"possession share exceeds 1 (max {worst:.3f}) for "
                         f"{len(bad)} player-seasons:\n{bad.head().to_string()}")

    games = pd.read_sql(
        "SELECT season, home_team_id t, COUNT(*) g FROM games "
        "GROUP BY season, home_team_id", con)
    # each team plays roughly twice its home count
    team_games = (games.assign(g=games.g * 2).groupby("season").g.mean()
                  .round().astype(int).to_dict())

    rapm = pd.read_sql(
        "SELECT season, player_id, player_name, net_p FROM rapm "
        "WHERE season != 'pooled'", con)

    df = rapm.merge(shares, on=["season", "player_id"], how="inner")
    df["team_games"] = df.season.map(team_games)
    df["vorp"] = ((df.net_p - REPLACEMENT_PER_100) * df.share
                  * df.team_games / FULL_SEASON_GAMES)
    df["war"] = df.vorp * WINS_PER_VORP

    # salaries (optional)
    seasons = sorted(df.season.unique())
    sal_by_season = {s: load_salaries(s) for s in seasons}
    have_salary = {s: bool(v) for s, v in sal_by_season.items()}
    df["salary"] = [sal_by_season[s].get(int(p))
                    for s, p in zip(df.season, df.player_id)]

    # cost per win, per season: total salary of players with a salary divided by
    # their summed wins over replacement. Only computable where salaries exist.
    cost_per_win = {}
    for s in seasons:
        g = df[(df.season == s) & df.salary.notna()]
        if len(g) and g.war.sum() > 0:
            cost_per_win[s] = float(g.salary.sum() / g.war.sum())
    df["value_usd"] = [
        (cost_per_win.get(s, float("nan")) * w) if s in cost_per_win else None
        for s, w in zip(df.season, df.war)]
    df["surplus_usd"] = [
        (v - sal) if (v is not None and sal is not None and pd.notna(v))
        else None
        for v, sal in zip(df.value_usd, df.salary)]

    df["teams"] = df.teams.apply(json.dumps)
    for c in ("share", "vorp", "war"):
        df[c] = df[c].round(4)
    keep = ["season", "player_id", "player_name", "teams", "poss", "share",
            "net_p", "vorp", "war", "salary", "value_usd", "surplus_usd"]
    df[keep].to_sql("player_value", con, if_exists="replace", index=False)

    meta = {
        "replacementPer100": REPLACEMENT_PER_100,
        "winsPerVorp": WINS_PER_VORP,
        "impactMetric": "rapm_net_prior",
        "teamGames": team_games,
        "salaryAvailable": have_salary,
        "costPerWin": cost_per_win,
    }
    con.execute("DROP TABLE IF EXISTS value_meta")
    con.execute("CREATE TABLE value_meta (json TEXT)")
    con.execute("INSERT INTO value_meta VALUES (?)", (json.dumps(meta),))
    con.commit()

    print(f"player_value: {len(df)} player-seasons")
    print(f"salary coverage: {have_salary}")
    top = df.sort_values("war", ascending=False).head(10)
    print("\ntop 10 by wins over replacement:")
    print(top[["season", "player_name", "poss", "net_p", "share", "war"]]
          .to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
