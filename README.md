# Over Expected

An end-to-end NBA shot-value system. It ingests six seasons of play-by-play,
trains leak-free models for what a shot is worth, and ships the results as a
data-journalism site.

**Live: [overexpected.com](https://overexpected.com)**

The core idea is a single question asked three ways: how much does a player add
over what an average player would do with the same looks?

### Where this repo fits

The live site now covers **nine leagues** and carries **four analytical layers**
on top of the three base lenses. It is split across four repositories:

| repo | what it is |
| --- | --- |
| **this one** | the NBA pipeline, and the backend for all four analytical layers |
| [over-expected-euro](https://github.com/quickstock/over-expected-euro) | the other eight leagues (EuroLeague, EuroCup, ACB, LBA, BBL, ABA, GBL, WNBA) through one adapter registry |
| [over-expected-site](https://github.com/quickstock/over-expected-site) | the React front end that ships to production, a pure consumer of exported JSON |
| [xpts-calibration](https://github.com/quickstock/xpts-calibration) | the validation record: reliability curves, nulls, and the metrics that were built and rejected |

The `site/` directory in *this* repo is the original NBA-only front end, kept as
the historical record. The site that actually deploys is `over-expected-site`.

- **Shot value**: points over expected per 100 possessions, fusing the shot and
  the fouls it draws.
- **Shot-making**: field-goal points over expected, actual conversion against
  the difficulty of the looks taken (xFG%).
- **Foul-drawing (FTAOE)**: shooting-foul free throws drawn over the league rate,
  the original stat the project is built around.

The same three lenses run on players, teams, and officials.

<!-- TODO: add screenshots: hero, a player page (the gap chart), the leaderboard, the crackdown trend -->

## Scale

- 1.39M possessions, 1.28M field-goal attempts, 7,230 games
- Six seasons (2020-21 to 2025-26), shooting fouls only
- ~550 statically prerendered routes with per-route OG cards and a sitemap

## How it works

```
nba_api  ->  cache/ (parquet)  ->  xfta.db (SQLite)  ->  models  ->  site/public/*.json  ->  React site
   pull.py        raw            possession + shot tables   LightGBM      export_site_data.py     Vite + TS
```

1. **Pull** (`pull.py`): network-only fetch of play-by-play, shot charts, box
   and tracking stats into a parquet cache. No table building here.
2. **Build**: possession-level and shot-level tables in `xfta.db`, including a
   target of shooting-foul free throws per possession.
3. **Model**:
   - Headline FTAOE: a possession-level model of expected shooting-foul free
     throws. FTAOE is actual minus expected, anchored per season so the league
     sits at zero and seasons are comparable.
   - xFG% (`xfg_model.py`): a LightGBM classifier giving every field-goal
     attempt a make probability from shot context only (location, distance,
     zone, action type, shot type, period, clock, score margin), never the
     shooter.
   - Shot value (`shot_value.py`): combines xFG% (make value) with expected
     free throws (foul value) into expected points per shot. The headline is
     points over expected per 100, crediting actual conversion on both sides
     (field goals at the player's rate, drawn free throws at his own FT%).
   - Style-adjusted FTAOE: a second baseline that predicts free throws from a
     player's attack profile (drives, paint and post touches), so the residual
     separates contact-seeking skill from sheer volume.
4. **Export** (`export_site_data.py`): writes a small core JSON plus per-season
   player chunks, validated by `scripts/validate_export.py`.
5. **Site** (`site/`): React + Vite + TypeScript, bespoke SVG charts, statically
   prerendered with OG cards for sharing.

## The analytical layers

Four layers sit on top of the three base lenses, each with its own pipeline
stages, its own versioned JSON, and its own stated ceiling. All four are NBA-only:
they need possession- and clock-level data no European feed publishes.

**Layer 1 — Lineups / RAPM** (`rapm_lib.py`, `build_stints.py`,
`train_rapm.py`, `build_lineups.py`). Offensive and defensive adjusted plus-minus
per player-season, plus five-man lineups measured against the sum of their parts.
Two things worth knowing: on-court state has to come from GameRotation, not
play-by-play, because the V3 feed never says who *started* a period. And a prior
you shrink toward must itself be shrunk — an unshrunk box-score prior made a
7-possession player the best offensive player in the league. RAPM beats
sum-of-parts out-of-sample in all six seasons; **lineup synergy does not persist**
(out-of-sample r = 0.02), and the board says so rather than selling it as
predictive.

**Layer 2 — Value** (`build_value.py`). Wins over replacement on the published
VORP shape, with this project's own adjusted plus-minus substituted for the
box-score estimate of the same quantity. Contract surplus is **not shipped**:
there is no free licensed per-player salary source, so the board says so instead
of rendering an invented dollar column.

**Layer 3 — Decision EV** (`build_etm.py`). The end-game two-versus-three choice,
scored on expected win probability at the moment of the decision. The scoring
function never receives the outcome; a test walks its syntax tree to prove it and
another re-scores every decision with the result flipped, asserting identical
answers. The finding: teams take the three about **55% of the time regardless of
margin**, while it is the higher-EV choice in 7.6% of situations trailing by one
and 99.8% trailing by three.

**Layer 4 — Team defence** (`build_defense.py`, `build_team_defense.py`).
Team-level on purpose. The player-level version was built, tested, and
**rejected**: 0.348 split-half reliability, and a leaderboard with Ja Morant among
the league's best defenders, because splitting every shot across five defenders
makes it team defence wearing a player's name. The team version was put through
the identical test before shipping — quality conceded 0.981, rim rate 0.964,
conversion suppression 0.717 — and the pillars ship ordered by measured
reliability so the noisy one cannot be read as the solid one. D-RAPM from Layer 1
remains the player defensive answer.

## Leak-free and honest by construction

The discipline is the point, not a footnote.

- **Leak-free season cross-fit.** For each season the models train on the other
  five and predict the held-out one, so a shot's expected value never comes from
  a model that saw it.
- **Anchored.** Within-season residuals sum to ~0, so seasons are directly
  comparable rather than drifting with the league's foul environment.
- **The possession is the unit.** A fouled miss is not a charged shot and has no
  location, so any per-shot rate silently drops the exact plays the stat is
  about. Rates are per 100 possessions.
- **Scoped claims.** The number blends playstyle, contact-seeking skill, and
  officiating. It does not isolate them and it does not prove referee bias. The
  site deliberately publishes no player-by-official splits, which on this sample
  size would manufacture accusations the data cannot support.
- **Validated.** `scripts/validate_export.py` is a gate: row counts, calibration,
  anchoring, the foul-ledger identity, and zone-share sums all have to pass
  before an export ships.

## A finding

The NBA's 2021-22 "non-basketball moves" crackdown barely moved the league rate
(17.8 to 17.5 shooting-foul FTA per 100). It was surgical: it repriced a handful
of high-volume foul-drawers rather than changing the whole game, and the
environment drifted back up the next season. The League tab draws this as a
season-by-season trend, with the full study at `/crackdown`.

## Repo layout

```
.
├── pull.py                  network pull -> cache/
├── build_*.py               possession + training tables
├── xfg_model.py             leak-free xFG% model
├── shot_value.py            shot value suite (xFG% + xFTA -> points)
├── backfill_score_margin.py reconstruct in-game margin from play-by-play
├── export_site_data.py      DB -> site JSON
├── scripts/validate_export.py  the export gate
├── config.py                seasons, feature lists, paths
│
│   # Layer 1 — lineups / RAPM
├── rapm_lib.py              pure helpers: clock, lineups, possession windows
├── build_stints.py          possession-level lineups + points
├── train_rapm.py            ridge with a shrunk box prior, sandwich SEs
├── build_lineups.py         five-man aggregates + synergy decomposition
├── export_rapm.py           -> rapm-NBA.json (+ per-season lineup logs)
│
│   # Layers 2-4
├── build_value.py           wins over replacement (surplus pending a source)
├── build_etm.py             end-game decision EV, outcome-blind by construction
├── build_defense.py         player-level defence — built, validated, REJECTED
├── build_team_defense.py    the team-level version the evidence supports
├── export_{value,etm,defense}.py   -> the layer JSON the site consumes
│
├── tests/                   contract, leakage, and hindsight-guard tests
└── site/                    the original NBA-only front end (superseded by
                             over-expected-site; kept as the record)
```

## Stack

- **Data and ML:** Python, pandas, SQLite, LightGBM, nba_api, numpy
- **Front end:** React, Vite, TypeScript, Tailwind, hand-built SVG charts, CSS
  motion with reduced-motion fallbacks
- **Build and deploy:** static prerender with satori OG cards, Vercel

## Running it

Data side (Python, from the repo root):

```bash
pip install -r requirements.txt   # nba_api, pandas, lightgbm, ...
python pull.py                    # fetch raw data to cache/ (network)
python shot_value.py              # train xFG%, build the shot-value tables
python export_site_data.py        # write site/public/*.json
python scripts/validate_export.py # gate
```

Site (from `site/`):

```bash
cd site
npm install
npm run dev          # local dev server
npm run build        # production build
```

## Credits

Built by Kevin Krajnc. Data from the NBA stats API. Shooting fouls only;
descriptive, not a referee-bias claim. See `/methodology` on the live site.
