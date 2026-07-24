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


def test_possession_windows_and_assignment():
    """Elapsed-time windows: (start_el, end_el], boundary events go to the
    EARLIER possession (the one they end); zero-width administrative
    possessions never capture events; cross-period carryovers (end clock
    ABOVE start clock) get an end in the following period and evaluate
    their lineup just after the break."""
    poss = pd.DataFrame({
        "possession_number": [1, 2, 3, 4, 5],
        "period":            [1, 1, 1, 2, 2],
        # Q1: 12:00->11:41 ; 11:41->0:03 ; carryover 0:03 -> Q2 11:50 ;
        # Q2: zero-width tech possession at 11:50 ; 11:50->11:30
        "start_time": ["PT12M00.00S", "PT11M41.00S", "PT00M03.00S",
                       "PT11M50.00S", "PT11M50.00S"],
        "end_time":   ["PT11M41.00S", "PT00M03.00S", "PT11M50.00S",
                       "PT11M50.00S", "PT11M30.00S"],
    })
    win = rl.possession_windows(poss)
    w3 = win[win.possession_number == 3].iloc[0]
    assert w3.start_el == 7170 and w3.end_el == 7200 + 100
    assert w3.lineup_eval_el == 7200 + rl.LINEUP_EPS      # after the break
    w1 = win[win.possession_number == 1].iloc[0]
    assert w1.lineup_eval_el == 0 + rl.LINEUP_EPS

    # events, in elapsed tenths:
    e_make_ends_p1 = 190            # poss 1 ends 11:41 -> elapsed 190
    e_boundary = 190                # same instant: belongs to poss 1
    e_carry_q2 = 7200 + 50          # carryover's Q2 segment (11:55)
    e_tech = 7200 + 100             # zero-width instant at Q2 11:50
    e_after = 7200 + 150            # inside possession 5
    got = rl.assign_possession_elapsed(
        np.array([0, e_make_ends_p1, e_boundary, e_carry_q2, e_tech,
                  e_after]), win)
    # the zero-width possession 4 must NOT capture the tech instant; the
    # carryover (3) ends exactly there, so boundary -> possession 3.
    assert got.tolist() == [1, 1, 1, 3, 3, 5]


def test_possession_windows_end_period_stamping():
    """The possession builder stamps SOME carryovers with the END period
    (start_time = previous period's dying instant), others with the START
    period. Cursor tiling must resolve both to the same elapsed windows."""
    poss = pd.DataFrame({
        "possession_number": [113, 114, 115],
        "period":            [2, 3, 3],
        "start_time": ["PT00M03.70S", "PT00M00.00S", "PT11M42.00S"],
        "end_time":   ["PT00M00.00S", "PT11M42.00S", "PT11M22.00S"],
    })
    win = rl.possession_windows(poss)
    w113 = win[win.possession_number == 113].iloc[0]
    w114 = win[win.possession_number == 114].iloc[0]
    w115 = win[win.possession_number == 115].iloc[0]
    assert w113.end_el == 14400                     # Q2 buzzer
    assert w114.start_el == 14400 and w114.end_el == 14580
    assert w114.lineup_eval_el == 14400 + rl.LINEUP_EPS
    assert w115.start_el == 14580 and w115.end_el == 14780
    # the Q3-opening make at 11:42 (elapsed 14580) belongs to 114
    got = rl.assign_possession_elapsed(np.array([14400, 14580, 14700]), win)
    assert got.tolist() == [113, 114, 115]


def test_possession_windows_eval_clamped_to_window():
    """A sub-second possession at a period's dying instant must evaluate
    its lineup INSIDE its own window, not EPS past the buzzer."""
    poss = pd.DataFrame({
        "possession_number": [1, 2],
        "period": [4, 4],
        "start_time": ["PT00M10.00S", "PT00M00.40S"],
        "end_time":   ["PT00M00.40S", "PT00M00.00S"],
    })
    win = rl.possession_windows(poss)
    w2 = win[win.possession_number == 2].iloc[0]
    assert w2.start_el == 28796 and w2.end_el == 28800
    # clamped inside its own window (end - 1), never past the buzzer
    assert w2.lineup_eval_el == 28799


def test_strip_suffix():
    assert rl.strip_suffix("bullock jr.") == "bullock"
    assert rl.strip_suffix("porter jr") == "porter"
    assert rl.strip_suffix("payton ii") == "payton"
    assert rl.strip_suffix("hardaway iii") == "hardaway"
    assert rl.strip_suffix("smith") == "smith"


def test_sub_timeline_suffix_resolution():
    """Description says 'Bullock', feed names say 'Bullock Jr.' — the
    suffix-stripped form must resolve."""
    rows = [
        (1, 1, "PT12M00.00S", 0, "", 0, "", "", "period", "start", ""),
        (2, 1, "PT11M00.00S", 1, "A", 10, "Bullock Jr.", "R. Bullock Jr.", "Made Shot", "", "Bullock 3PT"),
        (3, 1, "PT10M00.00S", 1, "A", 10, "Bullock Jr.", "R. Bullock Jr.", "Substitution", "", "SUB: Beta FOR Bullock"),
        (4, 1, "PT06M00.00S", 1, "A", 11, "Beta", "B. Beta", "Substitution", "", "SUB: Bullock FOR Beta"),
        (5, 1, "PT00M00.00S", 0, "", 0, "", "", "period", "end", ""),
    ]
    pbp = pd.DataFrame(rows, columns=[
        "actionNumber", "period", "clock", "teamId", "teamTricode",
        "personId", "playerName", "playerNameI", "actionType", "subType",
        "description"])
    rot, issues = rl.sub_timeline(pbp)
    assert issues == []
    b = rot[rot.PERSON_ID == 10].sort_values("IN_TIME_REAL")
    assert b.IN_TIME_REAL.tolist() == [0, 3600]
    assert b.OUT_TIME_REAL.tolist() == [1200, 7200]


def test_chrono_sort_relocates_late_inserted_rows():
    """v3 actionNumber is NOT chronological: amended events are re-appended
    with late numbers but correct clocks. Chronological order is
    (period, clock desc), stable so same-clock blocks keep feed order."""
    pbp = pd.DataFrame({
        "actionNumber": [10, 11, 12, 99, 13],
        "period":       [1, 1, 1, 1, 1],
        # 99 is a late-inserted make at 10:49, truly between 11:00 and 10:06
        "clock": ["PT11M00.00S", "PT10M06.00S", "PT09M56.00S",
                  "PT10M49.00S", "PT09M56.00S"],
        "actionType": ["Made Shot", "Foul", "Made Shot", "Made Shot",
                       "Free Throw"],
    })
    got = rl.chrono_sort(pbp).actionNumber.tolist()
    assert got == [10, 99, 11, 12, 13]   # 12 and 13 share 9:56, keep order


def test_event_points():
    pbp = pd.DataFrame({
        "actionNumber": [1, 2, 3, 4],
        "period":       [1, 1, 1, 1],
        "clock": ["PT11M00.00S", "PT10M30.00S", "PT10M00.00S", "PT10M00.00S"],
        "actionType": ["Made Shot", "Instant Replay", "Made Shot", "Free Throw"],
        "scoreHome": ["2", "0", "2", None],     # IR row carries garbage; FT unstamped
        "scoreAway": ["0", "9", "3", None],
    })
    d = rl.event_points(pbp)
    assert d.d_home.tolist() == [2, 0, 0, 0]
    assert d.d_away.tolist() == [0, 0, 3, 0]


def test_event_points_handles_out_of_order_amendments():
    """The 64-66 -> 66-66 -> 66-68 stream with the 64-66 make re-appended
    late must NOT read as a regression once chronologically sorted."""
    pbp = pd.DataFrame({
        "actionNumber": [366, 367, 368],
        "period":       [3, 3, 3],
        "clock": ["PT09M56.00S", "PT09M43.00S", "PT10M49.00S"],
        "actionType": ["Made Shot", "Made Shot", "Made Shot"],
        "scoreHome": ["66", "66", "64"],
        "scoreAway": ["66", "68", "66"],
    })
    d = rl.event_points(pbp)          # sorted: 368, 366, 367
    assert d.d_home.sum() == 66 and d.d_away.sum() == 68


def test_event_points_same_clock_score_order():
    """Within a same-clock block the feed's actionNumbers can lie (tech-FT
    between a trip's two FTs, putbacks numbered after later FTs). Scores
    are monotone in time, so stamped rows in a tie order by total score."""
    pbp = pd.DataFrame({
        "actionNumber": [359, 361, 364],
        "period": [2, 2, 2],
        "clock": ["PT00M03.30S"] * 3,
        "actionType": ["Free Throw", "Free Throw", "Free Throw"],
        # feed order: FT1 (46), tech FT (48), FT2 (47) -> true: 46,47,48
        "scoreHome": ["51", "51", "51"],
        "scoreAway": ["46", "48", "47"],
    })
    d = rl.event_points(pbp)          # must not raise
    assert d.d_away.sum() == 48 and (d.d_away >= 0).all()


def test_event_points_nulls_period_marker_scores():
    """Period start/end markers can stamp stale scores (a technical FT
    before the Q3 'Start' row, whose score predates the FT)."""
    pbp = pd.DataFrame({
        "actionNumber": [341, 349, 350],
        "period": [2, 3, 3],
        "clock": ["PT00M00.00S", "PT12M00.00S", "PT12M00.00S"],
        "actionType": ["period", "Free Throw", "period"],
        "scoreHome": ["46", "46", "46"],
        "scoreAway": ["31", "32", "31"],   # stale on the Q3 start marker
    })
    d = rl.event_points(pbp)          # must not raise
    assert d.d_away.sum() == 32


def test_event_points_regression_raises():
    # chronologically ordered rows whose score still decreases = spliced feed
    pbp = pd.DataFrame({
        "actionNumber": [1, 2],
        "period": [1, 1],
        "clock": ["PT11M00.00S", "PT10M00.00S"],
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


def test_sub_timeline_out_name_harvest():
    """The feed retroactively renames players (Kanter -> Freedom): fields
    say the new name while descriptions keep the old. Sub rows pair the
    OUT description-name with the row's personId — harvesting those pairs
    must let the old name resolve."""
    rows = [
        (1, 1, "PT12M00.00S", 0, "", 0, "", "", "period", "start", ""),
        (2, 1, "PT11M00.00S", 1, "A", 10, "Freedom", "E. Freedom", "Made Shot", "", "Kanter Layup"),
        (3, 1, "PT10M00.00S", 1, "A", 10, "Freedom", "E. Freedom", "Substitution", "", "SUB: Beta FOR Kanter"),
        (4, 1, "PT06M00.00S", 1, "A", 11, "Beta", "B. Beta", "Substitution", "", "SUB: Kanter FOR Beta"),
        (5, 1, "PT00M00.00S", 0, "", 0, "", "", "period", "end", ""),
    ]
    pbp = pd.DataFrame(rows, columns=[
        "actionNumber", "period", "clock", "teamId", "teamTricode",
        "personId", "playerName", "playerNameI", "actionType", "subType",
        "description"])
    rot, issues = rl.sub_timeline(pbp)
    assert issues == []
    k = rot[rot.PERSON_ID == 10].sort_values("IN_TIME_REAL")
    assert k.IN_TIME_REAL.tolist() == [0, 3600]
    assert k.OUT_TIME_REAL.tolist() == [1200, 7200]


def test_sub_timeline_elimination_resolution():
    """Two same-surname teammates and a bare-surname sub-in: the incoming
    player cannot already be on court, so elimination resolves it."""
    rows = [
        (1, 1, "PT12M00.00S", 0, "", 0, "", "", "period", "start", ""),
        (2, 1, "PT11M30.00S", 1, "A", 10, "Williams", "G. Williams", "Made Shot", "", "G. Williams 3PT"),
        (3, 1, "PT11M00.00S", 1, "A", 11, "Williams", "R. Williams", "Made Shot", "", "R. Williams Dunk"),
        # G. Williams (10) subs out; R. Williams (11) stays on court
        (4, 1, "PT10M00.00S", 1, "A", 10, "Williams", "G. Williams", "Substitution", "", "SUB: Beta FOR G. Williams"),
        # bare 'Williams' returns: must be 10 (11 is still on the floor)
        (5, 1, "PT06M00.00S", 1, "A", 12, "Beta", "B. Beta", "Substitution", "", "SUB: Williams FOR Beta"),
        (6, 1, "PT00M00.00S", 0, "", 0, "", "", "period", "end", ""),
    ]
    pbp = pd.DataFrame(rows, columns=[
        "actionNumber", "period", "clock", "teamId", "teamTricode",
        "personId", "playerName", "playerNameI", "actionType", "subType",
        "description"])
    rot, issues = rl.sub_timeline(pbp)
    assert issues == []
    g = rot[rot.PERSON_ID == 10].sort_values("IN_TIME_REAL")
    assert g.IN_TIME_REAL.tolist() == [0, 3600]
    assert g.OUT_TIME_REAL.tolist() == [1200, 7200]
    r = rot[rot.PERSON_ID == 11]
    assert len(r) == 1 and r.iloc[0].OUT_TIME_REAL == 7200


def test_sub_timeline_pending_in_retro_resolution():
    """Bare 'Williams' enters while BOTH Williamses are off the floor —
    elimination can't split them. His next action reveals which one it
    was; the stretch must backfill to the SUB time, not period start."""
    rows = [
        (1, 1, "PT12M00.00S", 0, "", 0, "", "", "period", "start", ""),
        (2, 1, "PT11M30.00S", 1, "A", 12, "Alpha", "A. Alpha", "Made Shot", "", "Alpha 3PT"),
        # both Williamses have acted earlier in the GAME (index has them)
        # but neither is on court in this period yet:
        (3, 1, "PT11M00.00S", 1, "A", 10, "Williams", "G. Williams", "Substitution", "", "SUB: Beta FOR G. Williams"),
        (4, 1, "PT10M30.00S", 1, "A", 11, "Williams", "R. Williams", "Substitution", "", "SUB: Gama FOR R. Williams"),
        (5, 1, "PT10M00.00S", 1, "A", 14, "Gama", "G. Gama", "Made Shot", "", "Gama Layup"),
        # ambiguous re-entry: both Williamses off court now
        (6, 1, "PT06M00.00S", 1, "A", 13, "Beta", "B. Beta", "Substitution", "", "SUB: Williams FOR Beta"),
        # ...revealed: G. Williams (10) scores
        (7, 1, "PT04M00.00S", 1, "A", 10, "Williams", "G. Williams", "Made Shot", "", "G. Williams Layup"),
        (8, 1, "PT00M00.00S", 0, "", 0, "", "", "period", "end", ""),
    ]
    pbp = pd.DataFrame(rows, columns=[
        "actionNumber", "period", "clock", "teamId", "teamTricode",
        "personId", "playerName", "playerNameI", "actionType", "subType",
        "description"])
    rot, issues = rl.sub_timeline(pbp)
    assert issues == []
    g = rot[rot.PERSON_ID == 10].sort_values("IN_TIME_REAL")
    # starter until 11:00 (600), back in at 6:00 (3600) via retro-resolution
    assert g.IN_TIME_REAL.tolist() == [0, 3600]
    assert g.OUT_TIME_REAL.tolist() == [600, 7200]


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
              test_lineup_at, test_possession_windows_and_assignment,
              test_possession_windows_end_period_stamping,
              test_possession_windows_eval_clamped_to_window,
              test_strip_suffix, test_sub_timeline_suffix_resolution,
              test_chrono_sort_relocates_late_inserted_rows,
              test_event_points,
              test_event_points_handles_out_of_order_amendments,
              test_event_points_same_clock_score_order,
              test_event_points_nulls_period_marker_scores,
              test_event_points_regression_raises, test_sub_timeline_basic,
              test_sub_timeline_reentry, test_sub_timeline_initial_name_resolution,
              test_sub_timeline_diacritics_resolution,
              test_sub_timeline_extra_index_fallback,
              test_sub_timeline_out_name_harvest,
              test_sub_timeline_elimination_resolution,
              test_sub_timeline_pending_in_retro_resolution,
              test_sub_timeline_unresolvable_flags_issue]:
        f()
        print(f"ok {f.__name__}")
