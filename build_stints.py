"""possessions + lineups + pbp -> possession_lineups (10 players, home
flag, points) per possession, with per-game gates. Skipped games recorded
in stint_skips with a reason; everything kept is exact by construction:
the sum of possession points must equal the final ffilled score on both
sides, and every possession must see exactly 5 players per team.

Lineup sources, per game:
  1. cache/rotations/{gid}.parquet (GameRotation) when present — the
     endpoint is aggressively throttled, so this fills slowly over time
     and repairs/audits the primary path;
  2. otherwise rapm_lib.sub_timeline() on the cached v3 play-by-play,
     with a season-level (team, name) -> id index (cached in
     cache/name_index/{season}.json) resolving sub-ins whose player never
     records an action in that game.

Lineup attribution: the 10 on court at possession start + 5 tenths
(start-of-possession rule; mid-possession FT-trip subs attribute the whole
possession to the starting lineup — documented in /methodology/lineups).
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd

import config
import rapm_lib as rl

ROT_DIR = os.path.join(config.CACHE_DIR, "rotations")
PBP_DIR = os.path.join(config.CACHE_DIR, "pbp")
IDX_DIR = os.path.join(config.CACHE_DIR, "name_index")


PLAYERS_DIR = os.path.join(config.CACHE_DIR, "players")


def _bio_forms(pid: int):
    """Name forms from the cached player bio: 'last', 'f. last',
    'fi. last' (the two-letter form the feed uses to split same-initial
    teammates like Co./Ca. Martin), 'first last' — each also
    suffix-stripped. Empty when no bio is cached."""
    path = os.path.join(PLAYERS_DIR, f"{pid}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            info = json.load(fh)["info"]
        first = (info.get("FIRST_NAME") or "").strip()
        last = (info.get("LAST_NAME") or "").strip()
    except (KeyError, ValueError):
        return []
    if not last:
        return []
    forms = {last}
    if first:
        forms.update({f"{first} {last}", f"{first[:1]}. {last}",
                      f"{first[:2]}. {last}"})
    out = set()
    for f in forms:
        n = rl.norm_name(f)
        out.add(n)
        out.add(rl.strip_suffix(n))
    return sorted(out)


def season_name_index(season: str, game_ids) -> dict:
    """{(teamId, norm_name): {personId}} across every game of a season,
    enriched with bio-derived name forms per player. Cached to JSON
    (keys 'tid|name'; cache version bumps when the derivation changes)."""
    path = os.path.join(IDX_DIR, f"{season}.v3.json")
    if os.path.exists(path):
        with open(path) as fh:
            raw = json.load(fh)
        return {(int(k.split("|", 1)[0]), k.split("|", 1)[1]): set(v)
                for k, v in raw.items()}
    idx: dict = {}
    team_pids: dict = {}

    def register(tid: int, form: str, pid: int):
        n = rl.norm_name(form)
        idx.setdefault((tid, n), set()).add(pid)
        s = rl.strip_suffix(n)
        if s != n:
            idx.setdefault((tid, s), set()).add(pid)

    for gid in game_ids:
        pbp_path = os.path.join(PBP_DIR, f"{gid}.parquet")
        if not os.path.exists(pbp_path):
            continue
        pbp = pd.read_parquet(
            pbp_path, columns=["teamId", "personId", "playerName",
                               "playerNameI", "actionType", "description"])
        pbp = pbp[(pbp.personId > 0) & (pbp.teamId > 0)]
        for tid, pid, n1, n2, at, desc in pbp.itertuples(index=False):
            tid, pid = int(tid), int(pid)
            team_pids.setdefault(tid, set()).add(pid)
            for form in (n1, n2):
                if form:
                    register(tid, form, pid)
            # contemporaneous OUT-name from sub descriptions (renames)
            if at == "Substitution":
                m = rl._SUB_RE.match(desc or "")
                if m:
                    register(tid, m.group(2), pid)
    for tid, pids in team_pids.items():
        for pid in pids:
            for n in _bio_forms(pid):
                idx.setdefault((tid, n), set()).add(pid)
    os.makedirs(IDX_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({f"{t}|{n}": sorted(ids) for (t, n), ids in idx.items()},
                  fh)
    return idx


def game_rotation(gid: str, pbp: pd.DataFrame, name_idx: dict):
    """(rot, source) — GameRotation parquet if cached, else sub-parsed."""
    path = os.path.join(ROT_DIR, f"{gid}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path), "rotation"
    rot, _issues = rl.sub_timeline(pbp, extra_index=name_idx)
    return rot, "subs"


def build_game(gid, season, home_id, away_id, poss, name_idx):
    pbp_path = os.path.join(PBP_DIR, f"{gid}.parquet")
    if not os.path.exists(pbp_path):
        return None, "pbp-missing", None
    pbp = pd.read_parquet(pbp_path)
    rot, source = game_rotation(gid, pbp, name_idx)

    windows = rl.possession_windows(poss)

    try:
        deltas = rl.event_points(pbp)
    except ValueError:
        return None, "score-regression", source
    ev = pbp.sort_values("actionNumber").copy()
    ev[["d_home", "d_away"]] = deltas
    ev["elapsed"] = [rl.elapsed_tenths(p, c)
                     for p, c in zip(ev.period, ev.clock)]
    ev["poss_no"] = rl.assign_possession_elapsed(
        ev.elapsed.to_numpy(), windows)

    pts_home = ev.groupby("poss_no").d_home.sum()
    pts_away = ev.groupby("poss_no").d_away.sum()
    inside_h = int(pts_home.drop(index=-1, errors="ignore").sum())
    inside_a = int(pts_away.drop(index=-1, errors="ignore").sum())
    if inside_h != int(ev.d_home.sum()) or inside_a != int(ev.d_away.sum()):
        return None, "points-identity", source

    # bulk on-court membership: rot rows x possessions
    ts = windows.lineup_eval_el.to_numpy()
    in_t = rot.IN_TIME_REAL.to_numpy()[:, None]
    out_t = rot.OUT_TIME_REAL.to_numpy()[:, None]
    on = (in_t <= ts[None, :]) & (ts[None, :] < out_t)
    team_arr = rot.TEAM_ID.to_numpy()
    pid_arr = rot.PERSON_ID.to_numpy().astype(int)

    rows = []
    for j, p in enumerate(poss.itertuples()):
        mask = on[:, j]
        home_pids = np.sort(pid_arr[mask & (team_arr == home_id)])
        away_pids = np.sort(pid_arr[mask & (team_arr == away_id)])
        if len(home_pids) != 5 or len(away_pids) != 5:
            return None, "coverage", source
        off_home = p.offense_team_id == home_id
        off = home_pids if off_home else away_pids
        deff = away_pids if off_home else home_pids
        pts = int((pts_home if off_home else pts_away)
                  .get(p.possession_number, 0))
        rows.append((gid, season, int(p.possession_number),
                     int(p.offense_team_id), int(off_home),
                     *off.tolist(), *deff.tolist(), pts))
    return rows, None, source


def main():
    con = sqlite3.connect(config.DB_PATH)
    games = pd.read_sql(
        "SELECT GAME_ID, season, home_team_id, away_team_id FROM games "
        "ORDER BY GAME_ID", con)
    poss_all = pd.read_sql(
        "SELECT game_id, period, possession_number, start_time, end_time, "
        "offense_team_id FROM possessions", con)
    by_game = dict(tuple(poss_all.groupby("game_id")))

    all_rows, skips, sources = [], [], {"rotation": 0, "subs": 0}
    for season, sgames in games.groupby("season"):
        name_idx = season_name_index(season, sgames.GAME_ID.tolist())
        for g in sgames.itertuples():
            poss = by_game.get(g.GAME_ID)
            if poss is None or not len(poss):
                skips.append((g.GAME_ID, "no-possessions"))
                continue
            rows, reason, source = build_game(
                g.GAME_ID, season, g.home_team_id, g.away_team_id,
                poss, name_idx)
            if source:
                sources[source] += 1
            if reason:
                skips.append((g.GAME_ID, reason))
            else:
                all_rows.extend(rows)
        print(f"[{season}] done ({len(all_rows)} rows so far, "
              f"{len(skips)} skips)", flush=True)

    cols = ["game_id", "season", "possession_number", "off_team", "home_off",
            "o1", "o2", "o3", "o4", "o5", "d1", "d2", "d3", "d4", "d5", "pts"]
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_sql("possession_lineups", con, if_exists="replace", index=False)
    sk = pd.DataFrame(skips, columns=["game_id", "reason"])
    sk.to_sql("stint_skips", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_pl_season "
                "ON possession_lineups(season)")
    con.commit()
    print(f"possession_lineups: {len(df)} rows, {df.game_id.nunique()} games")
    print(f"lineup sources: {sources}")
    print(f"skips ({len(sk)}): {sk.reason.value_counts().to_dict() if len(sk) else {}}")


if __name__ == "__main__":
    main()
