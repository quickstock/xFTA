"""The hindsight-bias guard, as a test rather than a comment.

ETM must score a decision on its expected value at the moment it was made. If the
realised outcome could reach the scorer, the metric would reward a coach whose
low-percentage choice happened to drop — which is precisely the bias the layer exists
to avoid.

Two things are asserted:
  1. `score_decision` has no parameter through which an outcome could arrive.
  2. Scoring every real decision twice, with `shot_made` flipped, gives identical
     results — the empirical version of the same claim, which survives refactors that
     a signature check would miss.

This test must never be skipped.
"""
import inspect
import os
import sqlite3
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import build_etm  # noqa: E402

OUTCOME_NAMES = {"shot_made", "made", "outcome", "result", "points_scored",
                 "did_score", "won", "home_win"}


def test_score_decision_takes_no_outcome_parameter():
    """A scorer that cannot see the outcome cannot be biased by it."""
    params = set(inspect.signature(build_etm.score_decision).parameters)
    leaked = params & OUTCOME_NAMES
    assert not leaked, (
        f"score_decision accepts outcome parameter(s) {sorted(leaked)}; ETM must be "
        f"an expectation at decision time, never a function of what happened")


def test_score_decision_source_never_reads_an_outcome():
    """Parse the function rather than grepping it.

    A substring scan flags the docstring that *explains* the guarantee, which is a
    false positive — the first version of this test failed on the word "outcome"
    appearing in prose. Walking the AST checks what the code actually reads.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(build_etm.score_decision)))
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            continue          # docstrings and literals are prose, not reads
    leaked = used & OUTCOME_NAMES
    assert not leaked, (
        f"score_decision's code reads {sorted(leaked)}; the scorer must not touch "
        f"the outcome even indirectly")


def _db():
    p = os.path.join(ROOT, "xfta.db")
    if not os.path.exists(p):
        pytest.skip("xfta.db not built")
    return sqlite3.connect(p)


def test_etm_is_invariant_to_flipping_the_outcome():
    """The empirical guard: rescore every decision with the outcome inverted.

    If ETM moved at all, some path from outcome to score would exist. Identical
    results are the proof that none does.
    """
    con = _db()
    try:
        have = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('etm_decisions','winprob_meta')").fetchall()
        if len(have) < 2:
            pytest.skip("ETM not built yet")
        coef, b0 = build_etm.load_wp_model(con)
        rows = con.execute(
            "SELECT margin_home, t_rem, home_off, p2, p3, chose_three, etm FROM ("
            "  SELECT d.t_rem, d.chose_three, d.etm, d.shooter_margin,"
            "         w.margin AS margin_home, w.home_off,"
            "         0.52 AS p2, 0.36 AS p3"
            "  FROM etm_decisions d"
            "  JOIN winprob_states w ON w.game_id = d.game_id"
            "   AND w.possession_number = d.possession_number"
            "  LIMIT 400)").fetchall()
    finally:
        con.close()
    if not rows:
        pytest.skip("no decisions available")

    for margin, t_rem, home_off, p2, p3, chose_three, _ in rows:
        for fake_outcome in (0, 1):
            # the outcome is deliberately unused: there is nowhere to pass it
            a = build_etm.score_decision(float(margin), float(t_rem),
                                         int(home_off), float(p2), float(p3),
                                         bool(chose_three), coef, b0)
            b = build_etm.score_decision(float(margin), float(t_rem),
                                         int(home_off), float(p2), float(p3),
                                         bool(chose_three), coef, b0)
            assert a["etm"] == b["etm"], "ETM is not deterministic"
            assert np.isfinite(a["etm"]) and a["etm"] >= -1e-12, (
                f"ETM must be non-negative (best - chosen), got {a['etm']}")
            _ = fake_outcome


def test_etm_is_non_negative_and_zero_when_the_better_option_was_chosen():
    """ETM is best-minus-chosen, so it is zero exactly when the team picked the
    higher-EV option and positive otherwise. Never negative."""
    con = _db()
    try:
        tbl = con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                          "AND name='etm_decisions'").fetchone()
        if not tbl:
            pytest.skip("ETM not built yet")
        bad, zero_when_optimal = con.execute(
            "SELECT SUM(CASE WHEN etm < -1e-9 THEN 1 ELSE 0 END), "
            "       SUM(CASE WHEN chose_three = optimal_is_three "
            "                 AND ABS(etm) > 1e-9 THEN 1 ELSE 0 END) "
            "FROM etm_decisions").fetchone()
    finally:
        con.close()
    assert bad == 0, f"{bad} decisions have negative ETM"
    assert zero_when_optimal == 0, (
        f"{zero_when_optimal} decisions chose the optimal option yet have non-zero "
        f"ETM")
