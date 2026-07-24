"""Lineup aggregates + synergy decomposition.

For each 5-man offensive lineup clearing a possession floor:
  net100     = 100 * (pts_for/poss_off - pts_against/poss_def)
  exp100     = sum of members' season prior-informed O-RAPM + D-RAPM
               (the sum-of-parts expectation; d is positive-is-good)
  synergy100 = net100 - exp100                         (the lineup effect)
  sq_synergy = lineup on-court xPts/shot  minus  the possession-weighted
               mean of members' individual season xPts/shot
               (does this five generate better LOOKS than its parts do)

sq_synergy reuses the offensive Over Expected primitive (shots_xfg), the
project's existing shot-quality model — the novel bit the platform spec
asks for. Defensive-synergy split is deferred to the defense layer and
flagged as such downstream.

OOS validation (spec requirement): split each season's games at the
midpoint date; correlate H1 synergy with H2 realized (net - expected) for
lineups with >= 120 poss in each half. Reported verbatim, whatever it is.
"""
import json
import sqlite3

import numpy as np
import pandas as pd

import config
import rapm_lib as rl

FLOOR_POSS = 300     # poss_off + poss_def, matching shot_value QUALIFY_POSS
HALF_FLOOR = 120     # per-half floor for the OOS check
OFF = ["o1", "o2", "o3", "o4", "o5"]
DEF = ["d1", "d2", "d3", "d4", "d5"]


def lineup_id(row, cols):
    return "-".join(str(int(row[c])) for c in cols)


def shot_lineups(con):
    """Map each xFG'd shot to the offensive lineup on the floor for its
    possession, reusing the tested elapsed-window assignment per game."""
    poss = pd.read_sql(
        "SELECT game_id, period, possession_number, start_time, end_time "
        "FROM possessions", con)
    pl = pd.read_sql(
        "SELECT game_id, possession_number, o1,o2,o3,o4,o5 "
        "FROM possession_lineups", con)
    shots = pd.read_sql(
        """SELECT s.game_id, s.event_id, s.period,
                  s.seconds_remaining_in_period AS secs, x.xpts
           FROM shots s JOIN shots_xfg x
             ON x.game_id = s.game_id AND x.event_id = s.event_id""", con)

    # vectorized elapsed time (period start + elapsed-within-period tenths)
    p_start = shots.period.map(rl.period_start_tenths).to_numpy()
    p_len = shots.period.map(rl.period_length_tenths).to_numpy()
    shots = shots.assign(
        elapsed=(p_start + p_len - shots.secs.to_numpy() * 10).astype(int))

    pos_frames = []
    poss_by_game = dict(tuple(poss.groupby("game_id")))
    for gid, sh in shots.groupby("game_id"):
        pg = poss_by_game.get(gid)
        if pg is None:
            continue
        win = rl.possession_windows(pg)
        pno = rl.assign_possession_elapsed(sh.elapsed.to_numpy(), win)
        pos_frames.append(pd.DataFrame({
            "game_id": gid, "possession_number": pno, "xpts": sh.xpts.values}))
    mapped = pd.concat(pos_frames, ignore_index=True)
    mapped = mapped[mapped.possession_number >= 0]
    # single vectorized join shot-possession -> offensive lineup
    pl = pl.copy()
    pl["lineup_id"] = pl[OFF].astype(int).astype(str).agg("-".join, axis=1)
    merged = mapped.merge(
        pl[["game_id", "possession_number", "lineup_id"]],
        on=["game_id", "possession_number"], how="inner")
    return merged[["game_id", "lineup_id", "xpts"]]


def main():
    con = sqlite3.connect(config.DB_PATH)
    pl = pd.read_sql("SELECT * FROM possession_lineups", con)
    games = pd.read_sql("SELECT GAME_ID, GAME_DATE, season FROM games", con)
    date_of = dict(zip(games.GAME_ID, games.GAME_DATE))
    rapm = pd.read_sql(
        "SELECT season, player_id, o_p, d_p, o, d FROM rapm", con)
    xps = pd.read_sql(
        "SELECT player_id, season, xpoints_per_shot FROM shot_value", con)

    pl["off_id"] = pl[OFF].astype(int).astype(str).agg("-".join, axis=1)
    pl["def_id"] = pl[DEF].astype(int).astype(str).agg("-".join, axis=1)

    # offense side: poss + points for
    off = (pl.groupby(["season", "off_id"])
             .agg(poss_off=("pts", "size"), pts_for=("pts", "sum"),
                  team=("off_team", lambda s: s.mode().iat[0]))
             .reset_index().rename(columns={"off_id": "lineup_id"}))
    # defense side: poss + points against (same 5 players when they DEFEND)
    deff = (pl.groupby(["season", "def_id"])
              .agg(poss_def=("pts", "size"), pts_against=("pts", "sum"))
              .reset_index().rename(columns={"def_id": "lineup_id"}))
    lu = off.merge(deff, on=["season", "lineup_id"], how="outer").fillna(
        {"poss_off": 0, "pts_for": 0, "poss_def": 0, "pts_against": 0})
    lu["poss"] = lu.poss_off + lu.poss_def
    lu = lu[lu.poss >= FLOOR_POSS].copy()
    lu["net100"] = 100 * (lu.pts_for / lu.poss_off.clip(lower=1)
                          - lu.pts_against / lu.poss_def.clip(lower=1))

    # expected = sum of members' prior-informed O-RAPM + D-RAPM
    rapm_o = rapm.set_index(["season", "player_id"]).o_p.to_dict()
    rapm_d = rapm.set_index(["season", "player_id"]).d_p.to_dict()
    xps_map = xps.set_index(["season", "player_id"]).xpoints_per_shot.to_dict()

    def members(lid):
        return [int(p) for p in lid.split("-")]

    exp, parts_xps = [], []
    for r in lu.itertuples():
        pids = members(r.lineup_id)
        exp.append(sum(rapm_o.get((r.season, p), 0.0)
                       + rapm_d.get((r.season, p), 0.0) for p in pids))
        xs = [xps_map.get((r.season, p)) for p in pids]
        xs = [v for v in xs if v is not None]
        parts_xps.append(float(np.mean(xs)) if xs else np.nan)
    lu["exp100"] = np.round(exp, 3)
    lu["synergy100"] = (lu.net100 - lu.exp100).round(3)
    lu["xpts_shot_parts"] = np.round(parts_xps, 4)

    # on-court shot quality per lineup
    sl = shot_lineups(con)
    oncourt = (sl.groupby("lineup_id").xpts.mean()
               .rename("xpts_shot").reset_index())
    # lineup_id in sl has no season; attach via the season-keyed lu rows
    lu = lu.merge(oncourt, on="lineup_id", how="left")
    lu["sq_synergy"] = (lu.xpts_shot - lu.xpts_shot_parts).round(4)
    lu["net100"] = lu.net100.round(3)

    # ---- per-game detail for the lineup pages
    lg = (pl.groupby(["season", "off_id", "game_id"])
            .agg(poss_off=("pts", "size"), pts_for=("pts", "sum"))
            .reset_index().rename(columns={"off_id": "lineup_id"}))
    lg_def = (pl.groupby(["season", "def_id", "game_id"])
                .agg(poss_def=("pts", "size"), pts_against=("pts", "sum"))
                .reset_index().rename(columns={"def_id": "lineup_id"}))
    lgames = lg.merge(lg_def, on=["season", "lineup_id", "game_id"],
                      how="outer").fillna(0)
    keep = set(zip(lu.season, lu.lineup_id))
    lgames = lgames[[(s, l) in keep for s, l in
                     zip(lgames.season, lgames.lineup_id)]]

    # ---- OOS: H1 synergy vs H2 realized (net - expected)
    oos = oos_check(pl, date_of, rapm_o, rapm_d)

    lu_cols = ["season", "lineup_id", "team", "poss_off", "poss_def",
               "pts_for", "pts_against", "poss", "net100", "exp100",
               "synergy100", "xpts_shot", "xpts_shot_parts", "sq_synergy"]
    lu[lu_cols].to_sql("lineup_season", con, if_exists="replace", index=False)
    lgames.to_sql("lineup_games", con, if_exists="replace", index=False)
    pd.DataFrame({"json": [json.dumps({"oos": oos, "floorPoss": FLOOR_POSS})]}
                 ).to_sql("lineup_meta", con, if_exists="replace", index=False)
    con.commit()

    print(f"lineup_season: {len(lu)} lineups "
          f"({lu.groupby('season').size().to_dict()})")
    print(f"OOS synergy->realized: r={oos['r']} n={oos['n']}")
    print("\ntop 8 synergy lineups (pooled across seasons):")
    show = lu.nlargest(8, "synergy100")[
        ["season", "team", "poss", "net100", "exp100", "synergy100",
         "sq_synergy"]]
    print(show.to_string(index=False))
    con.close()


def oos_check(pl, date_of, rapm_o, rapm_d):
    pl = pl.copy()
    pl["date"] = pl.game_id.map(date_of)   # "YYYY-MM-DD ..." sorts lexically
    rows = []
    for season, grp in pl.groupby("season"):
        dates = sorted(grp.date.unique())
        mid = dates[len(dates) // 2]        # median game-date, string order
        for half, sub in (("h1", grp[grp.date <= mid]),
                          ("h2", grp[grp.date > mid])):
            off = sub.groupby("off_id").agg(
                po=("pts", "size"), pf=("pts", "sum"))
            deff = sub.groupby("def_id").agg(
                pd_=("pts", "size"), pa=("pts", "sum"))
            m = off.join(deff, how="outer").fillna(0)
            m = m[(m.po + m.pd_) >= HALF_FLOOR]
            for lid, r in m.iterrows():
                net = 100 * (r.pf / max(r.po, 1) - r.pa / max(r.pd_, 1))
                exp = sum(rapm_o.get((season, int(p)), 0.0)
                          + rapm_d.get((season, int(p)), 0.0)
                          for p in lid.split("-"))
                rows.append((season, lid, half, net - exp, net))
    d = pd.DataFrame(rows, columns=["season", "lid", "half", "resid", "net"])
    piv = d.pivot_table(index=["season", "lid"], columns="half",
                        values="resid")
    piv = piv.dropna()
    if len(piv) < 10:
        return {"r": None, "n": int(len(piv)),
                "method": "H1 (net-exp) vs H2 (net-exp)"}
    r = float(np.corrcoef(piv.h1, piv.h2)[0, 1])
    return {"r": round(r, 3), "n": int(len(piv)),
            "method": "H1 (net-exp) vs H2 (net-exp)"}


if __name__ == "__main__":
    main()
