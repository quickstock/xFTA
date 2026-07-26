"""etm_team / etm_decisions -> site/public/coaching-NBA.json (Layer 3).

Decision expected-value for the end-game 2-vs-3 choice. One core file; the
per-decision table stays in the database because 1,364 rows of game state is
substrate for analysis, not something a page needs.

Two things this export deliberately reshapes rather than passing through:

**`distinguishable` is not the interesting question.** ETM is
WP(better option) − WP(chosen), so it is >= 0 by construction and every team's
interval excludes zero: all 30 are "distinguishably worse than always-optimal",
which is true but vacuous. The question a reader actually has is whether teams
separate from *each other*, and they barely do — the best team's interval ends
at 1.78pp and the worst's begins at 1.84pp. So this export computes overlap
tiers the same way the lineups layer does, and the board renders tiers.

**The margin breakdown is the finding**, not a footnote: optimality swings from
7.6% (trailing one) to 99.8% (trailing three) while choice sits near 55%
throughout. It ships as structured rows so the page cannot paraphrase it wrong.
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


def assign_tiers(rows):
    """Group teams whose intervals genuinely overlap into one tier.

    Same rule and same reasoning as the lineups layer: a ranked list of 30 teams
    implies distinctions these intervals do not support, so the constraint lives
    in the data rather than in the UI, where a styling change could quietly turn
    tiers back into ranks. Rows arrive best-first (lowest ETM).
    """
    tier, anchor_lo, anchor_hi = 1, None, None
    for row in rows:
        lo, hi = row["ci"]
        if anchor_lo is None:
            anchor_lo, anchor_hi = lo, hi
        elif lo > anchor_hi:            # no overlap with this tier's opener
            tier += 1
            anchor_lo, anchor_hi = lo, hi
        row["tier"] = tier
    return rows


def main():
    con = sqlite3.connect(config.DB_PATH)
    team = pd.read_sql("SELECT * FROM etm_team", con)
    dec = pd.read_sql("SELECT * FROM etm_decisions", con)
    emeta = json.loads(con.execute("SELECT json FROM etm_meta").fetchone()[0])
    con.close()

    # win probability reads far better in percentage points than in fractions
    pp = lambda v: round(float(v) * 100, 3)

    rows = []
    for r in team.sort_values("etm_mean").itertuples():
        rows.append({
            "team": TEAM_ABBREV.get(int(r.team_id), str(int(r.team_id))),
            "teamId": str(int(r.team_id)),
            "decisions": int(r.decisions),
            "etm": pp(r.etm_mean),
            "se": pp(r.se),
            "ci": [pp(r.ci_lo), pp(r.ci_hi)],
        })
    rows = assign_tiers(rows)

    margins = []
    for m in emeta["window"]["margins"]:
        g = dec[dec.shooter_margin == -m]
        if not len(g):
            continue
        margins.append({
            "margin": int(m),
            "n": int(len(g)),
            "optimalThreeShare": round(float(g.optimal_is_three.mean()), 4),
            "choseThreeShare": round(float(g.chose_three.mean()), 4),
        })

    # The spread across teams, so the page can state the range without
    # recomputing it and drifting from this file.
    best, worst = rows[0], rows[-1]

    core = {
        "meta": {
            "layer": "coaching", "league": "NBA", "version": VERSION,
            "generated": dt.datetime.now(dt.timezone.utc)
                           .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "unit": "percentage points of win probability lost per decision",
            "lowerIsBetter": True,
            "window": emeta["window"],
            "shotSeconds": emeta["shotSeconds"],
            "minDecisions": emeta["minDecisions"],
            "nDecisions": emeta["nDecisions"],
            "nTeams": emeta["nTeams"],
            "aggregationLevel": emeta["aggregationLevel"],
            "whyNotTeamSeason": emeta["whyNotTeamSeason"],
            "hindsightGuard": emeta["hindsightGuard"],
            "framing": emeta["framing"],
            "approximations": emeta["approximations"],
            "spread": {
                "bestTeam": best["team"], "bestEtm": best["etm"],
                "worstTeam": worst["team"], "worstEtm": worst["etm"],
                "tiers": max(r["tier"] for r in rows),
            },
            "distinguishableFromOptimal": {
                "share": emeta["distinguishableShare"],
                "note": ("ETM is >= 0 by construction, so every team's interval "
                         "excludes zero and this share is 100% by design. It is "
                         "reported for completeness, not as a finding — the "
                         "question that matters is whether teams separate from "
                         "each other, which is what the tiers answer."),
            },
            "tiering": {
                "rule": ("teams are sorted by ETM and grouped into contiguous "
                         "tiers; a team joins the current tier while its 95% "
                         "interval overlaps that of the team which opened it"),
                "why": ("with ~45 decisions per team the intervals are wide "
                        "enough that a 1-to-30 ranking would assert precision "
                        "the data does not have"),
            },
        },
        "margins": margins,
        "teams": rows,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "coaching-NBA.json"
    path.write_text(json.dumps(core, separators=(",", ":")))
    print(f"wrote coaching-NBA.json ({path.stat().st_size / 1e3:.1f} KB): "
          f"{len(rows)} teams in {core['meta']['spread']['tiers']} tiers, "
          f"{len(margins)} margin rows, {emeta['nDecisions']:,} decisions")
    print(f"  closest to optimal: {best['team']} {best['etm']}pp; "
          f"furthest: {worst['team']} {worst['etm']}pp")


if __name__ == "__main__":
    main()
