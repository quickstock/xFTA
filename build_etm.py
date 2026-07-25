"""Layer 3 (WO-10) — decision expected-value for the end-game 2-vs-3 choice.

ETM = WP(better option) − WP(option actually chosen), where both are evaluated on
**expectations at the moment of the decision**. A team down three with fifteen seconds
left either shoots a three, which can tie, or a two, which cannot. Which is better
depends on that team's own conversion rates and on what each branch does to its win
probability — never on whether the shot happened to go in.

**The hindsight-bias guard is the whole point of this file.** `score_decision` is
given the game state, the team's season rates, and which shot type was chosen. It is
never given the outcome, and `tests/test_hindsight_guard.py` proves that by scoring
every decision twice with `shot_made` flipped and asserting the results are identical.
A metric that rewards a coach for a shot that happened to drop is a metric that
rewards luck.

Framing, deliberately: this is "decision expected-value", not a judgement of a coach.
A negative ETM means a choice with lower expected win probability than the
alternative given that team's own rates — not that anyone is bad at their job. Where a
team's interval covers zero, the surface says so instead of ranking them.

Approximations, stated rather than buried:
  * The miss branch hands the ball to the opponent with the clock nearly expired,
    which is the common but not universal end-game case — offensive rebounds and
    intentional fouls are not modelled.
  * A shot is assumed to consume ~4 seconds from the decision.
  * Free throws and intentional-foul decisions are out of scope here; this file
    scores the shot-selection decision only.
"""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import rapm_lib as rl

MAX_SECS = 35.0          # end-game window
MARGINS = (1, 2, 3)      # trailing by one to three: the choice is live
SHOT_SECONDS = 4.0       # assumed time consumed by the attempt
MIN_DECISIONS = 25       # per TEAM (all seasons pooled), to report at all
# Team-season is too thin to report: 1,364 decisions over 30 teams x 6 seasons is
# about 8 per team-season, so intervals there are useless. Pooling seasons gives
# ~45 per team. That the per-season figure is unusable is itself worth stating.


def load_wp_model(con):
    meta = json.loads(con.execute("SELECT json FROM winprob_meta").fetchone()[0])
    c = meta["coef"]
    return (np.array([c["margin"], c["marginOverSqrtT"], c["logT"],
                      c["possession"]]), float(meta["intercept"]))


def wp_home(margin, t_rem, home_off, coef, b0):
    """P(home wins) from the calibrated end-game model."""
    t = max(float(t_rem), 1.0)
    x = np.array([margin, margin / np.sqrt(t), np.log1p(t), float(home_off)])
    return float(1.0 / (1.0 + np.exp(-(b0 + x @ coef))))


def score_decision(margin_home, t_rem, home_off, p2, p3, chose_three,
                   coef, b0):
    """EV of each branch, and the ETM of the choice made.

    Takes the state, the shooting team's own rates, and WHICH shot was chosen.
    It does not take the outcome, and must never be given one — that is what
    makes this a decision metric rather than a results metric.
    """
    sign = 1.0 if home_off else -1.0          # shooting team's points move home margin

    def wp_shooter(margin, t, off):
        w = wp_home(margin, t, off, coef, b0)
        return w if home_off else 1.0 - w

    t_after = max(t_rem - SHOT_SECONDS, 0.0)
    # make: shooting team keeps nothing (clock stops / opponent inbounds)
    ev3 = p3 * wp_shooter(margin_home + 3 * sign, t_after, not home_off) \
        + (1 - p3) * wp_shooter(margin_home, t_after, not home_off)
    ev2 = p2 * wp_shooter(margin_home + 2 * sign, t_after, not home_off) \
        + (1 - p2) * wp_shooter(margin_home, t_after, not home_off)
    chosen = ev3 if chose_three else ev2
    best = max(ev2, ev3)
    return {"ev2": ev2, "ev3": ev3, "chosen": chosen, "best": best,
            "etm": best - chosen,
            "optimal_is_three": bool(ev3 >= ev2)}


def first_shot_per_possession(con):
    """The shot that resolved each possession, attributed to its possession."""
    shots = pd.read_sql(
        """SELECT s.game_id, s.event_id, s.period,
                  s.seconds_remaining_in_period AS secs, s.team_id,
                  s.shot_made,
                  CASE WHEN s.shot_type LIKE '%3PT%' THEN 1 ELSE 0 END AS is3
           FROM shots s""", con)
    poss = pd.read_sql(
        "SELECT game_id, period, possession_number, start_time, end_time "
        "FROM possessions", con)
    out = []
    for gid, sg in shots.groupby("game_id", sort=False):
        pg = poss[poss.game_id == gid]
        if pg.empty:
            continue
        w = rl.possession_windows(pg)
        el = np.array([rl.elapsed_tenths(int(p), f"PT{int(s // 60):02d}M{s % 60:05.2f}S")
                       for p, s in zip(sg.period, sg.secs)], dtype=float)
        pn = rl.assign_possession_elapsed(el, w)
        sub = sg.assign(possession_number=pn, elapsed=el)
        out.append(sub[sub.possession_number >= 0])
    allshots = pd.concat(out, ignore_index=True)
    return (allshots.sort_values(["game_id", "possession_number", "elapsed"])
            .groupby(["game_id", "possession_number"], as_index=False).first())


def main():
    con = sqlite3.connect(config.DB_PATH)
    coef, b0 = load_wp_model(con)

    states = pd.read_sql(
        "SELECT game_id, season, possession_number, period, t_rem, margin, "
        "home_off FROM winprob_states WHERE period = 4 AND t_rem <= ?",
        con, params=(MAX_SECS,))
    print(f"end-game states (Q4, <= {MAX_SECS:.0f}s): {len(states):,}")

    shots = first_shot_per_possession(con)
    d = states.merge(shots[["game_id", "possession_number", "team_id",
                            "is3", "shot_made"]],
                     on=["game_id", "possession_number"], how="inner")

    # the decision is live only when the shooting team TRAILS by 1-3
    d["shooter_margin"] = np.where(d.home_off == 1, d.margin, -d.margin)
    d = d[d.shooter_margin.isin([-m for m in MARGINS])]
    print(f"live 2-vs-3 decisions (trailing by 1-3, shot attempted): {len(d):,}")

    # team-season conversion rates, the team's OWN rates per the spec
    rates = pd.read_sql(
        """SELECT g.season, s.team_id,
                  AVG(CASE WHEN s.shot_type LIKE '%3PT%' THEN s.shot_made END) p3,
                  AVG(CASE WHEN s.shot_type NOT LIKE '%3PT%' THEN s.shot_made END) p2
           FROM shots s JOIN games g ON g.GAME_ID = s.game_id
           GROUP BY g.season, s.team_id""", con)
    d = d.merge(rates, on=["season", "team_id"], how="left").dropna(
        subset=["p2", "p3"])

    rows = []
    for r in d.itertuples():
        sc = score_decision(float(r.margin), float(r.t_rem), int(r.home_off),
                            float(r.p2), float(r.p3), bool(r.is3), coef, b0)
        rows.append({"game_id": r.game_id, "season": r.season,
                     "possession_number": int(r.possession_number),
                     "team_id": int(r.team_id),
                     "shooter_margin": int(r.shooter_margin),
                     "t_rem": float(r.t_rem), "chose_three": bool(r.is3),
                     **{k: (float(v) if not isinstance(v, bool) else v)
                        for k, v in sc.items()}})
    dec = pd.DataFrame(rows)
    dec.to_sql("etm_decisions", con, if_exists="replace", index=False)

    print(f"\n=== how often is the three the higher-EV choice? ===")
    for m in MARGINS:
        g = dec[dec.shooter_margin == -m]
        if len(g) < 10:
            continue
        print(f"  trailing by {m}: three is optimal in {g.optimal_is_three.mean():.1%} "
              f"of {len(g):,} decisions; teams chose it {g.chose_three.mean():.1%} "
              f"of the time")

    # Team-level aggregate. Season-level is reported too, but only to show that it
    # is too thin to use rather than to be ranked on.
    per_season = dec.groupby(["season", "team_id"]).etm.size()
    print(f"\n  decisions per team-season: median {per_season.median():.0f}, "
          f"max {per_season.max()} -> too thin for intervals, so seasons are pooled")

    agg = dec.groupby(["team_id"]).agg(
        decisions=("etm", "size"), etm_mean=("etm", "mean"),
        etm_sd=("etm", "std")).reset_index()
    agg["se"] = agg.etm_sd / np.sqrt(agg.decisions)
    agg["ci_lo"] = agg.etm_mean - 1.96 * agg.se
    agg["ci_hi"] = agg.etm_mean + 1.96 * agg.se
    # a team is only distinguishable from optimal if its interval excludes zero
    agg["distinguishable"] = agg.ci_lo > 0
    agg = agg[agg.decisions >= MIN_DECISIONS]
    agg.to_sql("etm_team", con, if_exists="replace", index=False)

    print(f"\n=== teams with >= {MIN_DECISIONS} decisions (seasons pooled): "
          f"{len(agg)} ===")
    print(f"  mean ETM {agg.etm_mean.mean():.4f} win-probability per decision")
    print(f"  distinguishable from optimal (CI excludes 0): "
          f"{int(agg.distinguishable.sum())} of {len(agg)} "
          f"({agg.distinguishable.mean():.1%})")
    print("  -> where the interval covers zero the surface must say "
          "'not distinguishable',\n     not rank the team anyway.")

    meta = {
        "window": {"period": 4, "maxSeconds": MAX_SECS, "margins": list(MARGINS)},
        "shotSeconds": SHOT_SECONDS,
        "minDecisions": MIN_DECISIONS,
        "nDecisions": int(len(dec)),
        "nTeams": int(len(agg)),
        "aggregationLevel": "team, seasons pooled",
        "whyNotTeamSeason": ("about 8 decisions per team-season, so per-season "
                             "intervals are useless; pooling gives ~45 per team"),
        "distinguishableShare": round(float(agg.distinguishable.mean()), 4),
        "hindsightGuard": ("score_decision never receives shot_made; "
                           "tests/test_hindsight_guard.py flips the outcome on every "
                           "decision and asserts identical ETM"),
        "framing": ("decision expected-value, not a coach rating; negative ETM means "
                    "lower expected win probability than the alternative given that "
                    "team's own conversion rates"),
        "approximations": [
            "the miss branch hands the ball to the opponent with the clock nearly "
            "expired; offensive rebounds and intentional fouls are not modelled",
            f"a shot is assumed to consume {SHOT_SECONDS:.0f} seconds",
            "shot-selection only; intentional-foul and timeout decisions are not "
            "scored here",
            "no coach identity is available in this pipeline, so aggregation is by "
            "team-season rather than by coach",
        ],
    }
    con.execute("DROP TABLE IF EXISTS etm_meta")
    con.execute("CREATE TABLE etm_meta (json TEXT)")
    con.execute("INSERT INTO etm_meta VALUES (?)", (json.dumps(meta),))
    con.commit()
    con.close()
    top = agg.sort_values("etm_mean").head(3)
    bot = agg.sort_values("etm_mean", ascending=False).head(3)
    print("\n  closest to optimal:  " + ", ".join(
        f"{int(r.team_id)} {r.etm_mean:.4f}" for r in top.itertuples()))
    print("  furthest from optimal: " + ", ".join(
        f"{int(r.team_id)} {r.etm_mean:.4f}" for r in bot.itertuples()))
    print(f"\netm_decisions: {len(dec):,} rows; etm_team: {len(agg)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
