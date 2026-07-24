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


_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


def strip_suffix(norm: str) -> str:
    """Drop a trailing generational suffix from an already-normalized name
    ('bullock jr.' -> 'bullock'): sub descriptions omit them while the
    feed's playerName fields keep them."""
    parts = norm.split()
    if len(parts) > 1 and parts[-1] in _SUFFIXES:
        return " ".join(parts[:-1])
    return norm
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


LINEUP_EPS = 5  # tenths past the evaluation anchor when stamping lineups


def possession_windows(poss_df) -> pd.DataFrame:
    """Possession windows in ELAPSED tenths (monotone across the game).

    The possession builder lets possessions carry across period breaks with
    INCONSISTENT stamping: sometimes `period` is the start's period (start
    clock near 0:00 of that period, end clock in the next), sometimes the
    end's (start_time = the PREVIOUS period's dying instant). Per-row
    inference is therefore unreliable; instead, possessions tile the game,
    so each start is pinned to the previous end (cursor), and each end is
    the smallest elapsed candidate >= the start among the row's period
    interpretations {p-1, p, p+1}. The game-level points identity in
    build_stints is the loud backstop if a feed ever violates tiling.

    `lineup_eval_el` is where the on-court five are read: possession start
    for normal windows; for a window spanning a period break, just after
    the break (where the bulk of a carryover possession is played).

    Returns columns: possession_number, start_el, end_el, lineup_eval_el.
    """
    df = poss_df.sort_values("possession_number")
    end_t = df.end_time.map(clock_tenths).to_numpy()
    periods = df.period.astype(int).to_numpy()
    nums = df.possession_number.to_numpy()

    def elapsed_of(period: int, tenths_remaining: int) -> int:
        return (period_start_tenths(period)
                + period_length_tenths(period) - tenths_remaining)

    start_el = np.empty(len(df), dtype=np.int64)
    end_el = np.empty(len(df), dtype=np.int64)
    eval_el = np.empty(len(df), dtype=np.int64)
    # A sequence's first possession is never a carryover, so its stamped
    # (period, start_time) interpretation is trustworthy — and this keeps
    # partial frames (tests, diagnostics) valid too.
    cursor = elapsed_of(int(periods[0]),
                        clock_tenths(df.start_time.iloc[0]))
    for i, (p, et) in enumerate(zip(periods, end_t)):
        start_el[i] = cursor
        candidates = [elapsed_of(pp, et)
                      for pp in (p - 1, p, p + 1) if pp >= 1]
        end_el[i] = min(c for c in candidates if c >= cursor)
        # a boundary strictly inside the window -> carryover: read the
        # lineup right after the break. Clamp to end_el - 1 so a
        # sub-second possession at a period's dying instant never reads
        # the NEXT period's five (rotation stretches are [IN, OUT), so
        # t == end would already see the post-boundary lineup).
        boundaries = [period_start_tenths(pp)
                      for pp in range(2, periods.max() + 1)]
        inside = [b for b in boundaries if start_el[i] < b < end_el[i]]
        anchor = inside[-1] if inside else start_el[i]
        eval_el[i] = min(anchor + LINEUP_EPS, end_el[i] - 1)
        cursor = end_el[i]
    return pd.DataFrame({
        "possession_number": nums,
        "start_el": start_el,
        "end_el": end_el,
        "lineup_eval_el": eval_el,
    })


def assign_possession_elapsed(elapsed_arr, windows) -> np.ndarray:
    """Map event elapsed times to possession_number.

    Windows are (start_el, end_el]: a possession-ending make (and its
    same-instant and-1 / trip free throws) lands in the possession it
    finished, and a shared boundary always goes to the EARLIER possession.
    Zero-width administrative windows (start_el == end_el) never capture
    events. Events at or before the first possession's start (the opening
    jump) map to the first possession; anything else unmatched gets -1.
    """
    w = windows[windows.end_el > windows.start_el].sort_values("end_el")
    ends = w.end_el.to_numpy()
    starts = w.start_el.to_numpy()
    nums = w.possession_number.to_numpy()
    es = np.asarray(elapsed_arr, dtype=float)
    idx = np.searchsorted(ends, es, side="left")
    idx_c = np.clip(idx, 0, len(ends) - 1)
    valid = (idx < len(ends)) & (es > starts[idx_c])
    out = np.where(valid, nums[idx_c], -1)
    first = nums[np.argmin(starts)] if len(starts) else -1
    out = np.where(es <= (starts.min() if len(starts) else -1), first, out)
    return out.astype(np.int64)


def chrono_sort(pbp: pd.DataFrame) -> pd.DataFrame:
    """Chronological event order. v3's actionNumber is NOT chronological:
    amended/reviewed events are re-appended with late numbers but correct
    clocks. Sort by (period, clock desc) with a stable mergesort and
    actionNumber as the final key, so same-clock blocks (FT trips, and-1
    sequences) keep their feed order."""
    df = pbp.copy()
    df["_t"] = df.clock.map(clock_tenths)
    df = df.sort_values("actionNumber", kind="mergesort")
    df = df.sort_values(["period", "_t"], ascending=[True, False],
                        kind="mergesort")
    return df.drop(columns="_t")


def event_points(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-event score deltas from the running score. Returns d_home /
    d_away aligned to pbp's index (unstamped rows get 0).

    Untrusted score stamps are nulled first: Instant Replay rows carry
    stale pairs (verified on the WNBA feed, same infrastructure), and
    period start/end markers can repeat a pre-amendment score. The delta
    sequence is computed over the STAMPED rows only, ordered by (period,
    clock desc, total score asc, actionNumber): scores are monotone in
    real time, so ascending total is the true order inside a same-clock
    block, where the feed's actionNumbers are known to lie (tech FTs
    inserted mid-trip, putbacks renumbered after later FTs). A negative
    delta that survives that ordering means an amended row carries a
    stale CLOCK (score from the future at an earlier clock) — a spliced
    feed we cannot repair -> ValueError, caller skips the game.
    """
    df = pbp.copy()
    untrusted = df.actionType.isin(["Instant Replay", "period"])
    sh = pd.to_numeric(df.scoreHome.mask(untrusted), errors="coerce")
    sa = pd.to_numeric(df.scoreAway.mask(untrusted), errors="coerce")
    stamped = sh.notna() & sa.notna()

    sub = df[stamped].copy()
    sub["_sh"], sub["_sa"] = sh[stamped].astype(int), sa[stamped].astype(int)
    sub["_t"] = sub.clock.map(clock_tenths)
    sub["_total"] = sub._sh + sub._sa
    sub = sub.sort_values("actionNumber", kind="mergesort")
    sub = sub.sort_values(["period", "_t", "_total"],
                          ascending=[True, False, True], kind="mergesort")

    d_home = pd.Series(0, index=pbp.index, dtype=int)
    d_away = pd.Series(0, index=pbp.index, dtype=int)
    prev_h = prev_a = 0
    for i, h, a in zip(sub.index, sub._sh, sub._sa):
        dh, da = h - prev_h, a - prev_a
        if dh < 0 or da < 0:
            raise ValueError("score regression")
        d_home.at[i], d_away.at[i] = dh, da
        prev_h, prev_a = h, a
    return pd.DataFrame({"d_home": d_home, "d_away": d_away})


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
    df = chrono_sort(pbp)
    issues: list[str] = []

    # Name index per team, from every row that names a player (including
    # sub rows, whose personId/playerName belong to the OUT player). Keys
    # are diacritics-folded (sub descriptions are ASCII while playerName
    # fields keep the real spelling) and registered both with and without
    # generational suffixes ('Bullock Jr.' fields vs 'Bullock' in
    # descriptions).
    name_ids: dict[tuple, set] = {}

    def register(tid: int, form: str, pid: int):
        n = norm_name(form)
        name_ids.setdefault((tid, n), set()).add(pid)
        s = strip_suffix(n)
        if s != n:
            name_ids.setdefault((tid, s), set()).add(pid)

    for r in df.itertuples():
        pid = int(r.personId) if r.personId else 0
        tid = int(r.teamId) if r.teamId else 0
        if pid <= 0 or tid <= 0:
            continue
        for form in (r.playerName, r.playerNameI):
            if form:
                register(tid, form, pid)
        # Sub descriptions carry the CONTEMPORANEOUS name of the OUT
        # player (the row's personId) — the fields may hold a retroactive
        # rename (Kanter -> Freedom). Harvest the pairing.
        if r.actionType == "Substitution":
            m = _SUB_RE.match(r.description or "")
            if m:
                register(tid, m.group(2), pid)

    def resolve(tid: int, name: str, on_court=frozenset()):
        """Unique id for `name` on team `tid`; candidates already on the
        floor are eliminated (the incoming player cannot be on court)."""
        n = norm_name(name)
        for query in (n, strip_suffix(n)):
            for index in (name_ids, extra_index or {}):
                ids = index.get((tid, query), set())
                if len(ids) == 1:
                    return next(iter(ids))
                alive = ids - on_court
                if len(alive) == 1:
                    return next(iter(alive))
        return None

    stretches = []  # (team, pid, in_t, out_t)

    for period, grp in df.groupby("period", sort=True):
        p_start = period_start_tenths(int(period))
        p_end = p_start + period_length_tenths(int(period))
        on_since: dict = {}     # pid -> in_t
        team_of: dict = {}      # pid -> teamId
        outed: set = set()      # pids subbed out this period (and not back)
        pending: list = []      # ambiguous INs: [t, tid, frozenset(cands)]

        def try_pending(pid: int, tid: int):
            """A sighted player who is not on court may be the answer to
            an earlier ambiguous sub-in — backfill to the sub time."""
            for k, (t_in, p_tid, cands) in enumerate(pending):
                if p_tid == tid and pid in cands:
                    pending.pop(k)
                    return t_in
            return None

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
                    else:
                        t_in = try_pending(pid, tid)
                        if t_in is not None:      # retro-resolved entry
                            stretches.append((tid, pid, t_in, t))
                        elif pid not in outed:
                            # subbed out having never acted: period starter
                            stretches.append((tid, pid, p_start, t))
                    outed.add(pid)
                # IN: resolved by name within the same team; players
                # already on the floor are eliminated as candidates
                if m:
                    in_name = m.group(1).strip()
                    in_pid = resolve(tid, in_name,
                                     on_court=frozenset(on_since))
                    if in_pid is not None:
                        on_since[in_pid] = t
                        team_of[in_pid] = tid
                        outed.discard(in_pid)
                    else:
                        n = norm_name(in_name)
                        cands = set()
                        for q in (n, strip_suffix(n)):
                            for index in (name_ids, extra_index or {}):
                                cands |= index.get((tid, q), set())
                        cands -= set(on_since)
                        if cands:
                            # park it: the entrant's next action reveals
                            # which candidate came in
                            pending.append((t, tid, frozenset(cands)))
                        else:
                            issues.append(
                                f"p{period}: cannot resolve incoming "
                                f"'{in_name}' (team {tid})")
                continue

            if pid <= 0 or tid <= 0 or not _marks_presence(at, r.subType or ""):
                continue
            team_of.setdefault(pid, tid)
            if pid in on_since:
                continue
            t_in = try_pending(pid, tid)
            if t_in is not None:
                on_since[pid] = t_in
                outed.discard(pid)
                continue
            if pid in outed:
                issues.append(
                    f"p{period}: {r.playerName} acts after being subbed out")
                continue
            # first sighting this period without an entry sub: on since start
            on_since[pid] = p_start

        for t_in, tid, cands in pending:
            issues.append(
                f"p{period}: ambiguous sub-in at {t_in} never revealed "
                f"(team {tid}, {len(cands)} candidates)")
        for pid, t_in in on_since.items():
            stretches.append((team_of.get(pid, 0), pid, t_in, p_end))

    rot = pd.DataFrame(
        stretches, columns=["TEAM_ID", "PERSON_ID", "IN_TIME_REAL",
                            "OUT_TIME_REAL"])
    rot["IN_TIME_REAL"] = rot.IN_TIME_REAL.astype(float)
    rot["OUT_TIME_REAL"] = rot.OUT_TIME_REAL.astype(float)
    return rot, issues
