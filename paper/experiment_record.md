# Over Expected — verified experiment record (experiments_declared)

Provenance: all numbers below come from the repo's committed model artifacts and the
2026-07-01 pre-publication validation audit (this repo, commit f1a4b49). Nothing here
is estimated or to be re-derived. This file is the canonical experimental record for
the ARS pipeline intake.

## Data

- Six NBA regular seasons, 2020-21 through 2025-26. 7,230 games (1,080 in the
  COVID-shortened 2020-21, 1,230 each after).
- 1,282,312 field-goal attempts (shots table), 1,420,917 possessions.
- Sources: NBA Stats V3 live play-by-play + shotchartdetail (player-scoped, FGA
  context), both public endpoints. No tracking/optical data.
- Possession construction: pbpstats library splitter (no custom possession logic).

## Target definitions

- xFG target: shot_made (binary), one row per FGA.
- xFTA target: sfta = count of shooting-foul / and-1 free throws awarded in a
  possession (0-5; 1,278,429 zeros; 34,460 ones; 103,037 twos; 4,769 threes;
  213 fours; 9 fives). Technical, flagrant, clear-path, away-from-play, inbound
  and penalty FTs excluded. Total kept: 255,738 FTs; 67,569 excluded; 533
  contaminated trips dropped (misclassified and-1s, 0.085% audit rate).
- Identity sfta == ft_and1 + ft_sf2 + ft_sf3 holds on all 1,420,917 rows.
- Finisher attribution: FT shooter if a qualifying trip occurred, else last maker,
  else last turnover, else last shooter. 0% of fouled possessions lack a finisher;
  ~2% of all possessions do (dead possessions).

## Models (post-audit, current)

### xFG (expected field-goal %)
- LightGBM binary classifier: 700 trees, lr 0.03, 63 leaves, min_child 300,
  subsample 0.8, colsample 0.8, reg_lambda 1.0, seed 42.
- Features: shot_distance, angle_deg (atan2 of |x| vs y), shot_x, shot_y, period,
  seconds_remaining_in_period, run_margin (PRE-shot |score margin|: post-event
  margin minus the shot's own points), shot_zone_basic, shot_zone_area,
  action_type, shot_type.
- Isotonic calibration on a game-grouped 20% split (no game straddles fit/calib).
- Benchmark split: train 2022-23 + 2023-24 (fit 345,975 / calib 86,294),
  holdout 2024-25 (217,738 FGA, make rate 0.4675).
- Holdout metrics (honest, post-leak-fix): raw AUC 0.6578, Brier 0.22427,
  log-loss 0.63727, ECE 0.0096; calibrated AUC 0.6576, Brier 0.22422, ECE 0.0042.
- Player-evaluation predictions: 6-fold season cross-fit (train on 5 seasons,
  predict the 6th), then per-season anchoring (scale so each season's mean xFG
  equals that season's actual FG%). Anchor factors 1.013-1.025.

### xFTA (expected shooting-foul FTs per possession)
- Poisson GLM (statsmodels), possession grain, 6-fold season cross-fit, all
  predictions out-of-fold.
- Features: period, seconds_remaining_in_period, score_margin (at possession
  START), q4_or_ot (period >= 4 indicator; V3 PBP carries no bonus state),
  offense_is_home, opp_rate_logo (defensive team's season shooting-foul rate,
  leave-one-game-out), crew_rate_logo (mean of 3 assigned officials' season
  rates, leave-one-game-out).
- Season anchoring on the finisher universe (league FTAOE sums to ~0 per season;
  export gate enforces |poss-weighted mean per 100| <= 0.2).
- OOF anchored Poisson-deviance lift vs season-mean baseline: 0.195% global
  (folds: 0.185 / 0.168 / 0.170 / 0.254 / 0.244 / 0.152 for 2020-21..2025-26).
  Anchors 1.013-1.025.
- Player-season decile calibration on OOF aggregates is diagonal (slope ~1).

### Combined shot value (points over expected)
- FG side: actual FG points minus expected FG points (xFG x 2/3pt).
- FT side: actual shooting-foul FTA x player's own season FT% minus xFTA x the
  season's actual league FT% (0.7746-0.7843 by season, FTA-weighted).
- Qualification: >= 300 finisher-possessions (~280 players/season). 3,213
  qualified player-seasons in shot_value; 1,691 at the FTAOE board grain.

## Validation audit (2026-07-01) — what was checked, found, fixed

Checked: target definition walk-through; per-row target identities; duplicate keys
on every table; prediction coverage; season windows; train/test leakage (feature
timing + split construction); benchmark comparison to public league numbers;
elite/poor player direction checks; missing-data handling; cross-project constants.

Found and fixed (before -> after):
1. CRITICAL - xFG score-margin leak. shots.score_margin is stamped AFTER the
   event (verified against PBP), so run_margin contained the shot's own points.
   Controlled A/B retrain, identical split: leaky AUC 0.6705 -> clean 0.6578
   (-1.3 AUC pts of apparent skill was leakage). Calibration unaffected
   (ECE 0.0040 -> 0.0042).
2. MEANINGFUL - no xFG season anchoring. League-wide FG points over expected per
   100 ran -1.70 (2021-22) to +1.07 (2023-24); after anchoring, -0.25 to +0.42
   (residual = qualified-pool vs league-wide gap).
3. MEANINGFUL - flat 0.77 FT baseline vs actual 0.775-0.784 league FT%. League
   FT points over expected per 100 was +0.11..+0.32; now -0.02..+0.06.
4. REAL BUT IMMATERIAL - xFTA terminal-event margin leak (the stored possession
   margin included the predicted FTs; NULL on non-scoring possessions, so the
   filled feature half-encoded "possession scored"). Fixed to possession-start
   margin. Refit: max |FTAOE delta| 1.03 FTs across 1,691 qualified
   player-seasons, p99 0.73, zero top-20 membership changes in any season,
   max rank shift 5.
5. OPS - two deleted-but-required pipeline scripts restored (weekly update would
   have failed on next new-game week).
6. COSMETIC - stale calibration script read a superseded predictions table;
   "in_bonus" renamed q4_or_ot (it was period >= 4, never bonus state).

Headline effect of fixes: 2024-25 points-over-expected/100 leader flipped from
Jimmy Butler III (23.82, leaky) to Ty Jerome (22.99, clean); Butler 22.42.
FTAOE board effectively unchanged (Giannis 311.3 -> 311.8).

## Benchmark checks (post-fix, all pass)

League zone FG% by season vs public values: restricted area 0.641-0.671,
paint non-RA 0.427-0.446, mid-range 0.406-0.420, corner 3 0.383-0.397,
above-break 3 0.346-0.360, league eFG 0.531-0.546, league FT 0.775-0.784.

Direction checks 2024-25 (>=300 poss): shot-making top = Jokic +10.1pp,
Jerome +9.1, Durant +8.6, Pritchard +8.0; bottom = Mogbo -14.6, Capela -9.7.
FTAOE top = Giannis +311.8, Harden +192.5, Butler +190.7, SGA +175.0;
bottom = Bridges -120.0, Thompson -104.5, Pritchard -103.7, Hield -97.3.
Cross-season means: Curry shot-making +5.4pp with FTAOE -23; Embiid/Zion/SGA/
Giannis all large-positive FTAOE; Killian Hayes worst shot-maker (-4.9pp).

## Known residual limitations (to disclose)

- No tracking data: no defender distance, shot clock, touch time, or dribbles
  per shot (NBA exposes these only as per-player aggregates). Caps xFG
  discrimination (~0.66 AUC) well below tracking-era models; calibration, not
  discrimination, is the design goal.
- action_type describes the attempt ("Driving Dunk" vs "Jump Shot"); granularity
  may weakly correlate with outcome. Standard practice in public xFG models;
  accepted and disclosed.
- NBA 2025-26 heave rule: end-of-period heave misses are charged to the team,
  so 2025-26 player data has ~50 shots at 40+ ft (44% make) vs ~600 (2-4% make)
  in earlier seasons. Upstream accounting change; creates a cross-season
  discontinuity in long-range data.
- ~1,800 shots/season (2021-22..2024-25 only) carry no zone/location (PBP-only
  rows absent from shotchartdetail; league-average make rate) and are excluded
  from xFG. ~0.8% of shots.
- q4_or_ot is a crude proxy; true bonus state is not in the public V3 feed.
- sfta counts shooting-foul FTs only; penalty (non-shooting-foul bonus) FTs are
  out of scope by design.
- xFTA model lift over the season-mean baseline is small (~0.2% Poisson
  deviance) — possession-level foul events are mostly noise given public
  features; the metric's value is the anchored aggregation, not per-possession
  discrimination.
