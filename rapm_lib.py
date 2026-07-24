"""Pure helpers for the RAPM pipeline: game-clock parsing, elapsed-time
mapping (tenths, GameRotation units), lineups-at-instant, substitution-parsed
rotation timelines, event -> possession window assignment, and per-event
score deltas. No I/O here; everything unit-testable.

Two sources can produce the per-game rotation timeline (TEAM_ID, PERSON_ID,
IN_TIME_REAL, OUT_TIME_REAL):
  1. the GameRotation endpoint (authoritative, but aggressively throttled),
  2. sub_timeline() below, parsed offline from the cached v3 play-by-play
     (substitutions carry the OUT player's id but only the IN player's name,
     so names are resolved against the game's own action rows; period
     starters are inferred as players who act, or are subbed out, before
     entering — the classic public-data approach).
Both feed lineup_at() identically; build_stints gates every possession on a
full 5-on-5, so an under-inferred period fails loudly rather than silently.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")
_SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+?)\s*$")


def norm_name(name: str) -> str:
    """Fold to the feed's sub-description form: strip diacritics
    ('Valančiūnas' -> 'valanciunas'), straighten apostrophes, lowercase,
    collapse whitespace."""
    s = unicodedata.normalize("NFKD", str(name))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.replace("’", "'").lower()
    return " ".join(s.split())
REG_PERIOD_T = 7200   # tenths: 12 minutes
OT_PERIOD_T = 3000    # tenths: 5 minutes


def clock_tenths(clock):
    if not clock:
        return None
    m = _CLOCK_RE.match(clock)
    if not m:
        return None
    return int(round((int(m.group(1)) * 60 + float(m.group(2))) * 10))


def period_start_tenths(period: int) -> int:
    if period <= 4:
        return (period - 1) * REG_PERIOD_T
    return 4 * REG_PERIOD_T + (period - 5) * OT_PERIOD_T


def period_length_tenths(period: int) -> int:
    return REG_PERIOD_T if period <= 4 else OT_PERIOD_T


def elapsed_tenths(period: int, clock: str) -> int:
    return period_start_tenths(period) + (
        period_length_tenths(period) - clock_tenths(clock))


def lineup_at(rot: pd.DataFrame, t: int):
    """{TEAM_ID: sorted PERSON_IDs on court at elapsed t (IN <= t < OUT)}."""
    on = rot[(rot.IN_TIME_REAL <= t) & (rot.OUT_TIME_REAL > t)]
    return {
        team: tuple(sorted(grp.PERSON_ID.astype(int)))
        for team, grp in on.groupby("TEAM_ID")
    }


def assign_possession(period_arr, tenths_arr, poss_df) -> np.ndarray:
    """Map events (period, clock-tenths-remaining) to possession_number.

    Possession windows tile each period's clock. An event belongs to the
    window whose [end_t, start_t] contains it; a shared boundary
    (c == earlier possession's end == later possession's start) goes to the
    EARLIER possession, so possession-ending makes and their same-clock
    and-1 / trip free throws stay with the possession they finished.
    Events matching no window (period markers above the first start) get -1.
    """
    out = np.full(len(period_arr), -1, dtype=np.int64)
    for p, g in poss_df.groupby("period"):
        g = g.sort_values("end_t")
        ends = g.end_t.to_numpy()
        starts = g.start_t.to_numpy()
        nums = g.possession_number.to_numpy()
        mask = np.asarray(period_arr) == p
        cs = np.asarray(tenths_arr, dtype=float)[mask]
        idx = np.searchsorted(ends, cs, side="right") - 1
        valid = (idx >= 0) & (cs <= starts[np.clip(idx, 0, len(starts) - 1)])
        res = np.where(valid, nums[np.clip(idx, 0, len(nums) - 1)], -1)
        out[mask] = res
    return out


def event_points(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-event score deltas from the running score. Instant Replay rows
    stamp stale score pairs on this API family (verified on the WNBA feed,
    same infrastructure) and are nulled before the ffill. Any negative
    delta means a spliced/duplicated feed -> ValueError, caller skips."""
    df = pbp.sort_values("actionNumber").copy()
    ir = df.actionType == "Instant Replay"
    out = {}
    for col, name in (("scoreHome", "d_home"), ("scoreAway", "d_away")):
        s = pd.to_numeric(df[col].mask(ir), errors="coerce")
        s = s.ffill().fillna(0).astype(int)
        d = s.diff().fillna(s.iloc[0]).astype(int)
        if (d < 0).any():
            raise ValueError("score regression")
        out[name] = d
    return pd.DataFrame(out, index=df.index)


# ------------------------------------------------------- sub timelines

# Rows that must never create on-court presence: dead-ball administration,
# and fouls charged to the bench (technical) which regularly name players
# who are not in the game.
_NO_PRESENCE = {"Substitution", "Timeout", "period", "Instant Replay",
                "Ejection", ""}


def _marks_presence(action_type: str, sub_type: str) -> bool:
    if action_type in _NO_PRESENCE:
        return False
    if action_type == "Foul" and "technical" in (sub_type or "").lower():
        return False
    return True


def sub_timeline(pbp: pd.DataFrame, extra_index=None):
    """Rotation timeline parsed from v3 pbp substitutions + presence.

    Returns (rot, issues): rot has GameRotation's shape (TEAM_ID, PERSON_ID,
    IN_TIME_REAL, OUT_TIME_REAL, floats); issues is a list of human-readable
    anomalies (unresolvable incoming names, post-out reappearances). The
    caller decides whether an issue is fatal — build_stints' 5-on-5 gate is
    the real arbiter.

    `extra_index` — optional {(teamId, norm_name): {personId, ...}} fallback
    built at season level; needed for the player whose ONLY trace in a game
    is a sub-in he never followed with a recorded action (his id then
    appears nowhere in this game's rows).
    """
    df = pbp.sort_values(["period", "actionNumber"])
    issues: list[str] = []

    # Name index per team, from every row that names a player (including
    # sub rows, whose personId/playerName belong to the OUT player). Keys
    # are diacritics-folded: sub descriptions are ASCII while playerName
    # fields keep the real spelling.
    name_ids: dict[tuple, set] = {}
    for r in df.itertuples():
        pid = int(r.personId) if r.personId else 0
        tid = int(r.teamId) if r.teamId else 0
        if pid <= 0 or tid <= 0:
            continue
        for form in (r.playerName, r.playerNameI):
            if form:
                name_ids.setdefault((tid, norm_name(form)), set()).add(pid)

    def resolve(tid: int, name: str):
        key = (tid, norm_name(name))
        for index in (name_ids, extra_index or {}):
            ids = index.get(key, set())
            if len(ids) == 1:
                return next(iter(ids))
            if len(ids) > 1:
                return None  # ambiguous even in initialed form
        return None

    stretches = []  # (team, pid, in_t, out_t)

    for period, grp in df.groupby("period", sort=True):
        p_start = period_start_tenths(int(period))
        p_end = p_start + period_length_tenths(int(period))
        on_since: dict = {}     # pid -> in_t
        team_of: dict = {}      # pid -> teamId
        outed: set = set()      # pids subbed out this period (and not back)

        for r in grp.itertuples():
            at = r.actionType or ""
            pid = int(r.personId) if r.personId else 0
            tid = int(r.teamId) if r.teamId else 0

            if at == "Substitution":
                t = elapsed_tenths(int(r.period), r.clock)
                m = _SUB_RE.match(r.description or "")
                # OUT: the row's personId
                if pid > 0:
                    if pid in on_since:
                        stretches.append((tid, pid, on_since.pop(pid), t))
                    elif pid not in outed:
                        # subbed out having never acted: period starter
                        stretches.append((tid, pid, p_start, t))
                    outed.add(pid)
                # IN: resolved by name within the same team
                if m:
                    in_name = m.group(1).strip()
                    in_pid = resolve(tid, in_name)
                    if in_pid is None:
                        issues.append(
                            f"p{period}: cannot resolve incoming "
                            f"'{in_name}' (team {tid})")
                    else:
                        on_since[in_pid] = t
                        team_of[in_pid] = tid
                        outed.discard(in_pid)
                continue

            if pid <= 0 or tid <= 0 or not _marks_presence(at, r.subType or ""):
                continue
            team_of.setdefault(pid, tid)
            if pid in on_since:
                continue
            if pid in outed:
                issues.append(
                    f"p{period}: {r.playerName} acts after being subbed out")
                continue
            # first sighting this period without an entry sub: on since start
            on_since[pid] = p_start

        for pid, t_in in on_since.items():
            stretches.append((team_of.get(pid, 0), pid, t_in, p_end))

    rot = pd.DataFrame(
        stretches, columns=["TEAM_ID", "PERSON_ID", "IN_TIME_REAL",
                            "OUT_TIME_REAL"])
    rot["IN_TIME_REAL"] = rot.IN_TIME_REAL.astype(float)
    rot["OUT_TIME_REAL"] = rot.OUT_TIME_REAL.astype(float)
    return rot, issues
