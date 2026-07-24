"""Unit tests for rapm_lib pure functions. Run directly or via pytest."""
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rapm_lib as rl


def test_clock_tenths():
    assert rl.clock_tenths("PT06M43.50S") == 4035
    assert rl.clock_tenths("PT12M00.00S") == 7200
    assert rl.clock_tenths("PT00M00.00S") == 0
    assert rl.clock_tenths("") is None
    assert rl.clock_tenths(None) is None


def test_period_start_tenths():
    assert rl.period_start_tenths(1) == 0
    assert rl.period_start_tenths(4) == 21600
    assert rl.period_start_tenths(5) == 28800   # OT1
    assert rl.period_start_tenths(6) == 31800   # OT2


def test_elapsed_tenths():
    assert rl.elapsed_tenths(1, "PT12M00.00S") == 0
    assert rl.elapsed_tenths(1, "PT06M43.00S") == 3170
    assert rl.elapsed_tenths(4, "PT00M00.00S") == 28800
    assert rl.elapsed_tenths(5, "PT05M00.00S") == 28800
    assert rl.elapsed_tenths(5, "PT00M00.00S") == 31800


def test_lineup_at():
    rot = pd.DataFrame({
        "TEAM_ID":   [1, 1, 2],
        "PERSON_ID": [10, 11, 20],
        "IN_TIME_REAL":  [0.0, 3170.0, 0.0],
        "OUT_TIME_REAL": [3170.0, 7200.0, 7200.0],
    })
    assert rl.lineup_at(rot, 5)[1] == (10,)     # in [0, 3170)
    assert rl.lineup_at(rot, 3170)[1] == (11,)  # boundary: out at 3170, in at 3170
    assert rl.lineup_at(rot, 3170)[2] == (20,)


def test_assign_possession():
    # possession 1: Q1 12:00 -> 11:41 ; possession 2: 11:41 -> 11:23
    poss = pd.DataFrame({
        "possession_number": [1, 2],
        "period": [1, 1],
        "start_t": [7200, 7010],
        "end_t": [7010, 6830],
    })
    periods = np.array([1, 1, 1, 1])
    tenths = np.array([7200, 7010, 6900, 6830])
    got = rl.assign_possession(periods, tenths, poss)
    # 7200 = poss 1 start (start boundary belongs to the possession it
    # opens ONLY when no earlier possession ends there; here poss 1's
    # window is [7010, 7200] closed at the period-opening start);
    # 7010 = shared boundary -> earlier possession (1);
    # 6900 inside poss 2; 6830 = poss 2 end -> poss 2.
    assert got.tolist() == [1, 1, 2, 2]


def test_event_points():
    pbp = pd.DataFrame({
        "actionNumber": [1, 2, 3, 4],
        "actionType": ["Made Shot", "Instant Replay", "Made Shot", "Free Throw"],
        "scoreHome": ["2", "0", "2", None],     # IR row carries garbage; FT unstamped
        "scoreAway": ["0", "9", "3", None],
    })
    d = rl.event_points(pbp)
    assert d.d_home.tolist() == [2, 0, 0, 0]
    assert d.d_away.tolist() == [0, 0, 3, 0]


def test_event_points_regression_raises():
    pbp = pd.DataFrame({
        "actionNumber": [1, 2],
        "actionType": ["Made Shot", "Made Shot"],
        "scoreHome": ["2", "1"], "scoreAway": ["0", "0"],
    })
    try:
        rl.event_points(pbp)
        assert False, "should raise"
    except ValueError:
        pass


def _mini_pbp():
    """Synthetic v3-shaped pbp: one 12-min period, two teams (tri A/B).

    Team A (teamId 1): players 10 (Alpha), 11 (Beta), 12 (Gama) — 10 and 11
    start; 11 is subbed out for 12 at 6:00 having never recorded an action
    (starter inferred from the OUT row); 12 scores later.
    Team B (teamId 2): 20 (Delta) starts and plays through; 21 (Echo) is a
    bench player who commits a TECHNICAL foul at 8:00 (must NOT create
    presence); 22 (Zeta) enters at 3:00 for Delta.
    """
    rows = [
        # actionNumber, period, clock, teamId, tri, personId, playerName, playerNameI, actionType, subType, description
        (1, 1, "PT12M00.00S", 0, "", 0, "", "", "period", "start", "Start of 1st Period"),
        (2, 1, "PT11M30.00S", 1, "A", 10, "Alpha", "A. Alpha", "Made Shot", "Jump Shot", "Alpha 15' Jump Shot"),
        (3, 1, "PT08M00.00S", 2, "B", 21, "Echo", "E. Echo", "Foul", "Technical", "Echo Technical (bench)"),
        (4, 1, "PT06M00.00S", 1, "A", 11, "Beta", "B. Beta", "Substitution", "", "SUB: Gama FOR Beta"),
        (5, 1, "PT05M00.00S", 1, "A", 12, "Gama", "G. Gama", "Made Shot", "Layup Shot", "Gama 2' Layup"),
        (6, 1, "PT04M00.00S", 2, "B", 20, "Delta", "D. Delta", "Missed Shot", "Jump Shot", "MISS Delta 20' Jump Shot"),
        (7, 1, "PT03M00.00S", 2, "B", 20, "Delta", "D. Delta", "Substitution", "", "SUB: Zeta FOR Delta"),
        (8, 1, "PT01M00.00S", 2, "B", 22, "Zeta", "Z. Zeta", "Made Shot", "Layup Shot", "Zeta 2' Layup"),
        (9, 1, "PT00M00.00S", 0, "", 0, "", "", "period", "end", "End of 1st Period"),
    ]
    return pd.DataFrame(rows, columns=[
        "actionNumber", "period", "clock", "teamId", "teamTricode",
        "personId", "playerName", "playerNameI", "actionType", "subType",
        "description"])


def test_sub_timeline_basic():
    rot, issues = rl.sub_timeline(_mini_pbp())
    assert issues == []
    # Beta (11): starter with zero actions, out at 6:00 elapsed 3600
    beta = rot[rot.PERSON_ID == 11]
    assert len(beta) == 1
    assert beta.iloc[0].IN_TIME_REAL == 0 and beta.iloc[0].OUT_TIME_REAL == 3600
    # Gama (12): in at 3600, plays to period end 7200
    gama = rot[rot.PERSON_ID == 12]
    assert gama.iloc[0].IN_TIME_REAL == 3600 and gama.iloc[0].OUT_TIME_REAL == 7200
    # Echo (21): bench technical only -> NO presence row
    assert (rot.PERSON_ID == 21).sum() == 0
    # Delta (20): starter, out at elapsed 5400 (3:00 remaining -> 9:00 played)
    delta = rot[rot.PERSON_ID == 20]
    assert delta.iloc[0].IN_TIME_REAL == 0 and delta.iloc[0].OUT_TIME_REAL == 5400
    # lineup_at mid-period agrees
    lu = rl.lineup_at(rot, 3000)
    assert lu[1] == (10, 11) and lu[2] == (20,)


def test_sub_timeline_reentry():
    """Player out and back in within the same period -> two stretches."""
    rows = [
        (1, 1, "PT12M00.00S", 0, "", 0, "", "", "period", "start", ""),
        (2, 1, "PT10M00.00S", 1, "A", 10, "Alpha", "A. Alpha", "Substitution", "", "SUB: Beta FOR Alpha"),
        (3, 1, "PT09M00.00S", 1, "A", 11, "Beta", "B. Beta", "Made Shot", "Layup Shot", "Beta 2' Layup"),
        (4, 1, "PT05M00.00S", 1, "A", 11, "Beta", "B. Beta", "Substitution", "", "SUB: Alpha FOR Beta"),
        (5, 1, "PT00M00.00S", 0, "", 0, "", "", "period", "end", ""),
    ]
    pbp = pd.DataFrame(rows, columns=[
        "actionNumber", "period", "clock", "teamId", "teamTricode",
        "personId", "playerName", "playerNameI", "actionType", "subType",
        "description"])
    rot, issues = rl.sub_timeline(pbp)
    assert issues == []
    alpha = rot[rot.PERSON_ID == 10].sort_values("IN_TIME_REAL")
    assert alpha.IN_TIME_REAL.tolist() == [0, 4200]
    assert alpha.OUT_TIME_REAL.tolist() == [1200, 7200]


def test_sub_timeline_initial_name_resolution():
    """Two same-surname teammates: description uses the initialed form."""
    rows = [
        (1, 1, "PT12M00.00S", 0, "", 0, "", "", "period", "start", ""),
        (2, 1, "PT11M00.00S", 1, "A", 10, "Smith", "J. Smith", "Made Shot", "", "J. Smith 3PT"),
        (3, 1, "PT10M00.00S", 1, "A", 11, "Smith", "K. Smith", "Made Shot", "", "K. Smith Layup"),
        (4, 1, "PT06M00.00S", 1, "A", 10, "Smith", "J. Smith", "Substitution", "", "SUB: Gama FOR J. Smith"),
        (5, 1, "PT03M00.00S", 1, "A", 12, "Gama", "G. Gama", "Substitution", "", "SUB: K. Smith FOR Gama"),
        (6, 1, "PT00M00.00S", 0, "", 0, "", "", "period", "end", ""),
    ]
    # wait: K. Smith is already on court at 3:00 (never left). Entering again
    # is a feed impossibility -> use a third player as the re-entrant instead.
    rows[4] = (5, 1, "PT03M00.00S", 1, "A", 12, "Gama", "G. Gama", "Substitution", "", "SUB: J. Smith FOR Gama")
    pbp = pd.DataFrame(rows, columns=[
        "actionNumber", "period", "clock", "teamId", "teamTricode",
        "personId", "playerName", "playerNameI", "actionType", "subType",
        "description"])
    rot, issues = rl.sub_timeline(pbp)
    assert issues == []
    j = rot[rot.PERSON_ID == 10].sort_values("IN_TIME_REAL")
    assert j.IN_TIME_REAL.tolist() == [0, 5400]
    assert j.OUT_TIME_REAL.tolist() == [3600, 7200]


def test_sub_timeline_diacritics_resolution():
    """The feed ASCII-folds names in sub descriptions ('Valanciunas') while
    playerName keeps diacritics ('Valančiūnas') — resolution must match."""
    rows = [
        (1, 1, "PT12M00.00S", 0, "", 0, "", "", "period", "start", ""),
        (2, 1, "PT11M00.00S", 1, "A", 10, "Jović", "N. Jović", "Made Shot", "", "Jovic 3PT"),
        (3, 1, "PT10M00.00S", 1, "A", 11, "Alpha", "A. Alpha", "Substitution", "", "SUB: Beta FOR Alpha"),
        (4, 1, "PT06M00.00S", 1, "A", 12, "Beta", "B. Beta", "Substitution", "", "SUB: Jovic FOR Beta"),
        (5, 1, "PT00M00.00S", 0, "", 0, "", "", "period", "end", ""),
    ]
    pbp = pd.DataFrame(rows, columns=[
        "actionNumber", "period", "clock", "teamId", "teamTricode",
        "personId", "playerName", "playerNameI", "actionType", "subType",
        "description"])
    # Jović acts (starter), is never subbed out before 6:00... he IS on
    # court from the start; the sub at 6:00 brings him ON again which is a
    # feed impossibility — restructure: he must be OUT first. Simpler: he
    # starts, gets subbed out at 10:00, re-enters at 6:00 via the folded name.
    rows = [
        (1, 1, "PT12M00.00S", 0, "", 0, "", "", "period", "start", ""),
        (2, 1, "PT11M00.00S", 1, "A", 10, "Jović", "N. Jović", "Made Shot", "", "Jovic 3PT"),
        (3, 1, "PT10M00.00S", 1, "A", 10, "Jović", "N. Jović", "Substitution", "", "SUB: Beta FOR Jovic"),
        (4, 1, "PT06M00.00S", 1, "A", 11, "Beta", "B. Beta", "Substitution", "", "SUB: Jovic FOR Beta"),
        (5, 1, "PT00M00.00S", 0, "", 0, "", "", "period", "end", ""),
    ]
    pbp = pd.DataFrame(rows, columns=pbp.columns)
    rot, issues = rl.sub_timeline(pbp)
    assert issues == []
    j = rot[rot.PERSON_ID == 10].sort_values("IN_TIME_REAL")
    assert j.IN_TIME_REAL.tolist() == [0, 3600]
    assert j.OUT_TIME_REAL.tolist() == [1200, 7200]


def test_sub_timeline_extra_index_fallback():
    """A player with no rows at all in this game (garbage-time sub-in who
    never acts) resolves through the caller-supplied season index."""
    rows = [
        (1, 1, "PT12M00.00S", 0, "", 0, "", "", "period", "start", ""),
        (2, 1, "PT06M00.00S", 1, "A", 10, "Alpha", "A. Alpha", "Substitution", "", "SUB: Ghost FOR Alpha"),
        (3, 1, "PT00M00.00S", 0, "", 0, "", "", "period", "end", ""),
    ]
    pbp = pd.DataFrame(rows, columns=[
        "actionNumber", "period", "clock", "teamId", "teamTricode",
        "personId", "playerName", "playerNameI", "actionType", "subType",
        "description"])
    extra = {(1, "ghost"): {99}}
    rot, issues = rl.sub_timeline(pbp, extra_index=extra)
    assert issues == []
    g = rot[rot.PERSON_ID == 99]
    assert g.iloc[0].IN_TIME_REAL == 3600 and g.iloc[0].OUT_TIME_REAL == 7200


def test_sub_timeline_unresolvable_flags_issue():
    """Incoming name matching nothing -> issue recorded, no crash."""
    rows = [
        (1, 1, "PT12M00.00S", 0, "", 0, "", "", "period", "start", ""),
        (2, 1, "PT06M00.00S", 1, "A", 10, "Alpha", "A. Alpha", "Substitution", "", "SUB: Ghost FOR Alpha"),
        (3, 1, "PT00M00.00S", 0, "", 0, "", "", "period", "end", ""),
    ]
    pbp = pd.DataFrame(rows, columns=[
        "actionNumber", "period", "clock", "teamId", "teamTricode",
        "personId", "playerName", "playerNameI", "actionType", "subType",
        "description"])
    rot, issues = rl.sub_timeline(pbp)
    assert any("Ghost" in s for s in issues)


if __name__ == "__main__":
    for f in [test_clock_tenths, test_period_start_tenths, test_elapsed_tenths,
              test_lineup_at, test_assign_possession, test_event_points,
              test_event_points_regression_raises, test_sub_timeline_basic,
              test_sub_timeline_reentry, test_sub_timeline_initial_name_resolution,
              test_sub_timeline_diacritics_resolution,
              test_sub_timeline_extra_index_fallback,
              test_sub_timeline_unresolvable_flags_issue]:
        f()
        print(f"ok {f.__name__}")
