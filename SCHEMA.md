# Database schema (`xfta.db`)

SQLite. Six seasons (2020-21 to 2025-26), 7,230 games, shooting fouls only.
Tables fall into three groups: ingested base data, training features, and model
outputs that feed the site.

## Rebuild order

```
pull.py                       network pull -> cache/ (parquet)
build_tables.py               -> games, shots, training_fga, player_season
build_possessions_v3.py       -> possessions  (per-possession shooting-foul target)
build_training_possessions_v2.py -> training_possessions_v2 (possession features)
backfill_score_margin.py      -> fills shots.score_margin from play-by-play
train_possession_v4_context.py-> predictions_poss_clean  (expected FTA per possession)
build_possession_leaderboard_clean.py -> player_season_xfta_poss_lb_clean (FTAOE board)
build_style_adjusted.py       -> style_expected  (attack-profile baseline)
build_player_ft.py            -> player_season_ft  (season FT%)
xfg_model.py / shot_value.py  -> shot_value, player_game_shot_value, team_shot_value,
                                 shots_xfg (per-shot OOF xFG/xPts)
export_site_data.py           -> site/public/*.json   (gated by scripts/validate_export.py)

# Layer 1 — lineups / RAPM (independent of the FTAOE chain above)
pull_rotations.py             network pull -> cache/rotations/{gid}.parquet
build_stints.py               -> possession_lineups, stint_skips
train_rapm.py                 -> rapm, rapm_meta
build_lineups.py              -> lineup_season, lineup_games, lineup_meta
export_rapm.py                -> site/public/rapm-NBA.json + lineups-NBA-{season}.json
                                 (gated by scripts/validate_rapm.py)
```

## Base / ingested

### games (7,230)
`GAME_ID, season, GAME_DATE, home_team_id, away_team_id`. One row per regular
season game.

### game_meta (7,230)
`game_id, home_team_id, away_team_id, ref1_id, ref1_name, ref2_id, ref2_name,
ref3_id, ref3_name, n_officials`. Officiating crew per game, source for the
referee profiles.

### shots (1,282,312)
One row per field-goal attempt.
`game_id, event_id, player_id, team_id, period, seconds_remaining_in_period,
shot_made, shot_x, shot_y, shot_distance, shot_zone_basic, shot_zone_area,
shot_zone_range, action_type, shot_type, home_or_away, score_margin`.
`event_id` maps 1:1 to play-by-play `actionNumber`. `score_margin` is signed by
the shooting team and is now complete (backfilled from play-by-play; about 5%
are genuine ties). A fouled miss is not a charged shot and is absent here, which
is why the foul-drawing stat is per possession, not per shot.

### possessions (1,420,917)
One row per possession.
`game_id, period, possession_number, start_time, end_time, offense_team_id,
n_events, sfta, finisher_player_id, excluded_ft_count, contamination_count,
ft_and1, ft_sf2, ft_sf3`. `sfta` is the shooting-foul free throw target (almost
always 0; 1-3 on a foul trip, rarely 4-5 when offensive rebounds produce two
foul trips in one possession). `ft_and1 / ft_sf2 / ft_sf3` itemize free throws
by trip type and sum to `sfta`. The finisher is the player charged with the
possession's shooting-foul free throws.

### and1_shots (34,454)
`game_id, event_id, player_id, possession_number`. Made shots that drew a foul
(the only shooting fouls with an official shot location), used for the
foul-origin court view.

### player_season (6,288)
`player_id, player_name, height_inches, position, season, prior_season_ftr,
prior_season_drive_rate, possessions`. Player attributes plus prior-season rates
used as carried features.

### tracking_exposures (3,407)
`player_id, gp, drives, paint_touches, post_touches, season`. Attack profile
feeding the style-adjusted baseline.

## Training features

### training_fga (1,282,312)
The modeling superset, one row per FGA, joining shot context to the foul target.
`game_id, event_id, player_id, season, shot_distance, shot_zone_basic,
shot_zone_area, action_type, shot_type, period, seconds_remaining_in_period,
score_margin, in_bonus, home_or_away, shooter_height, shooter_position,
prior_season_ftr, prior_season_drive_rate, fta_from_shot`. The site also reads
this for per-player charged-FGA shot zones.

## Model outputs

### predictions_poss_clean (1,420,917)
`game_id, possession_number, season, sfta, xfta`. The headline FTAOE model's
out-of-fold expected free throws per possession, leak-free season cross-fit
(written by `train_possession_v4_context.py`). FTAOE = `sfta - xfta`.

### player_season_xfta_poss_lb_clean (3,566)
The FTAOE leaderboard, one row per qualified player-season.
`player_id, season, possessions, actual_fta_from_fouls, xfta_total, ftaoe,
ftaoe_per_100, ftaoe_rank, player_name, position`. Anchored per season so the
league sits at zero and seasons are comparable.

### style_expected (1,691)
`player_id, season, style_xfta`. Expected free throws from a player's attack
profile alone (drives, paint and post touches). The residual against actual is
the style-adjusted FTAOE shown on player pages.

### player_season_ft (3,407)
`player_id, player_name, season, ftm, fta, ft_pct`. Season free-throw rate, used
to value a player's drawn free throws at his own line in the shot-value headline.

### shot_value (3,213)
The shot-value suite, one row per qualified player-season.
`player_id, player_name, season, position, possessions, fga, fgm, fg_pct,
xfg_pct, shot_making_oe, xpoints_per_shot, exp_fg_pts, act_fg_pts, fg_pts_oe,
fg_pts_oe_per100, actual_fta, xfta_total, ftaoe, ft_pct, ft_pts_oe,
ft_pts_oe_per100, points_oe, points_oe_per100`. `xfg_pct` is the leak-free
expected FG% for the looks taken; `points_oe_per100` is the headline (field-goal
points over expected plus free throws drawn at the player's own FT%).

### player_game_shot_value (145,131)
`player_id, game_id, season, fga, fgm, act_fg_pts, exp_fg_pts`. Per-game FG
points actual vs expected, the series behind the shot-value gap and form charts.

### team_shot_value (180)
`team_id, season, off_* / def_* (fga, act_fg_pts, exp_fg_pts)`. Team FG points
over expected on both ends, retained for analysis (the live League board uses
the foul-drawing side).

## Layer 1 — lineups / RAPM

### possession_lineups (1,382,146)
One row per possession with both five-man units resolved.
`game_id, season, possession_number, off_team, home_off, o1..o5, d1..d5, pts`.
On-court state comes from `cache/rotations` (GameRotation), not from
play-by-play substitutions — the V3 feed never states who *started* a period, so
on/off can't be reconstructed from it alone. Coverage is 97.3% of possessions
across 7,040 of 7,230 games; `pts` is reconstructed from the raw feed (made FG
`shotValue` + made FTs) and gated against each game's official final.

### stint_skips (190)
`game_id, reason`. Games dropped whole rather than half-modeled: `coverage` (162,
rotation rows don't resolve to 5-on-5 everywhere) and `score-regression` (28, the
same corrupt-feed signal `build_tables` uses).

### rapm (4,491) / rapm_meta
One row per player-season plus a pooled fit.
`player_id, player_name, season, poss_off, poss_def, o, d, net, o_p, d_p, net_p,
se_o, se_d`. Bare fields are plain ridge; `*_p` shrink toward a
leave-one-season-out box-score prior instead of toward zero, and are what the
site displays. Defence is signed so positive prevents points. λ is chosen per
season by game-holdout CV (3,200 — an interior optimum of the grid, not a
boundary), and RAPM beats both a sum-of-parts and a home-court-only baseline
out-of-sample in every season. `rapm_meta` holds that CV table, the prior's R²
(~0.34 offence, ~0.10 defence — box stats predict offence far better), the
collinearity report (no pairs above r=0.95), and skip counts.

**The prior is itself shrunk by sample size** (`PRIOR_SHRINK_POSS`). It is a
linear fit on per-100 box rates, so a player with a handful of possessions has
wild per-100 rates and the fit extrapolates absurdly; because the ridge shrinks
the *residual* toward the prior, a garbage prior becomes the estimate. Unshrunk,
a 7-possession player priced at +14.8 O-RAPM and topped the board.

### lineup_season (889) / lineup_games (15,760) / lineup_meta
Five-man units at or above a 300-possession floor.
`season, lineup_id, team, poss_off, poss_def, pts_for, pts_against, poss,
net100, exp100, synergy100, xpts_shot, xpts_shot_parts, sq_synergy`.
`exp100` sums members' prior-informed RAPM, so `synergy100 = net100 - exp100` is
performance beyond the sum of the parts; `sq_synergy` asks the same question of
shot quality using `shots_xfg`. **Synergy does not persist out-of-sample**
(r=0.02, n=321), so the site presents it as description, never forecast.

### shots_xfg (1,274,964)
`game_id, event_id, xfg, xpts`. Per-shot out-of-fold xFG and expected points,
persisted so lineup-level shot quality can reuse the shot-value primitive rather
than refit it.

## Notes

- The numbers are descriptive. They blend playstyle, contact-seeking skill, and
  officiating, and do not isolate or prove referee bias.
- Models are leak-free (season cross-fit) and anchored per season.
- `xfta.db` is gitignored; a slim gzip is kept in version control.
