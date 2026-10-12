# Design: architecture, decisions, weaknesses

## Architecture

```
                 ┌──────────────────────────────┐      ┌──────────────────────────────┐
 GitHub raw CSV  │ data.py                      │      │ telegram.py                  │  t.me/s/kpszsu
 (volunteer,  ──►│ load_alerts / truncate_at    │      │ scrape_channel (incremental, │◄── (public preview)
 daily dump)     └──────────────┬───────────────┘      │ backfill)                    │
                                │ (region,start,end)   └──────────────┬───────────────┘
                                ▼                                     │ (post_id, published_at, text)
                 ┌──────────────────────────────┐                     │
                 │ features.Grid                │   geo.py: adjacency, │
                 │  minute matrix  active[R,M]  │   UA regexes ───────►│
                 │  starts per minute / hour    │                     ▼
                 │  occupancy per hour          │      ┌──────────────────────────────┐
                 └──────────────┬───────────────┘      │ telegram_hourly / _recent    │
                                ▼                      └──────────────┬───────────────┘
                 ┌──────────────────────────────────────────────────────┐
                 │ features.build_frame: one row per (hour t, oblast)   │
                 │  features from [.., t)  |  targets y_h from [t, t+h) │
                 └──────────────┬───────────────────────────────────────┘
                                ▼
 scripts/run_experiment.py:  train | val (28 d) | test (28 d), h-hour embargo
   1. training-window selection (val)      4. refit train+val → test once
   2. Telegram ablation, 3 arms (val)      5. weekly walk-forward refit → test
   3. hyper-parameter grid (val)           6. day-block bootstrap CIs
                                ▼
 models/lgb_{1,3,6}h.txt + meta.json ──► scripts/predict.py (live table, P1≤P3≤P6)
 reports/*.csv ──► scripts/make_report.py ──► reports/RESULTS.md, figures/
 tests/test_leakage.py: features(full data)[t ≤ cut] == features(data truncated at cut)
```

One **global** LightGBM per horizon (all oblasts in one model, oblast as a categorical feature) rather than
24 local models: rare-alert oblasts (Zakarpattia: ~1 onset/week) borrow strength from the rest, and cross-region
signals (neighbours, national activity) are shared.

## Key decisions and why

| decision | alternatives considered | why |
|---|---|---|
| Target = **onset of a new alert** within 1/3/6 h | "alert active", duration regression | "Active" is trivial via persistence while an alert is on; onset is what a person planning the next hours needs. Chosen with the user. |
| **Volunteer** oblast-level dataset | official dataset | Official file stops 2026-09-07 and is raion-level in 2026; the task requires the latest data. |
| Hourly forecast origins, **minute** raster for features | hourly bins only | Minute raster makes "running alert, its age, time since last alert" exact and makes the leakage rule trivially checkable. |
| **Leakage test** instead of reasoning only | code review | Recomputing features on truncated data is a mechanical proof; it also covers Telegram posts. |
| Train / val / test by time with **h-hour embargo** | random CV, TimeSeriesSplit | Rows are heavily correlated in time and across oblasts; random CV would overstate quality. |
| Baselines: climatology, seasonal naive, logistic regression | none | A model is only meaningful against what a simple rule achieves. |
| LightGBM | deep sequence models (LSTM/TCN), Prophet/ARIMA per region | Tabular features + GBM is strong, fast (minutes on CPU), easy to inspect; per-region ARIMA cannot use cross-region signals; deep models are not justified by ~1 year of relevant data. |
| Telegram via **regex** on oblast names, declensions, towns | LLM/NER extraction, embeddings | Transparent, fast, no API cost; errors were found and fixed by inspecting real matches. |
| Telegram arm chosen on **validation**, bootstrap on test | eyeballing test AUC | Avoids selecting on test; CI shows whether a +0.01 AUC gain is real. |
| Weekly walk-forward refit | static model | Measured aging of the static model across test weeks. |
| Enforce P1h ≤ P3h ≤ P6h at prediction time | leave independent models | Logical consistency for the user; violations were rare (<1%) but up to 0.2. |

## Weaknesses and risks (honest list)

1. **Not truly real-time.** The alert dump is refreshed about daily (last record 04:59 UTC on 12 Oct), so
   `predict.py` forecasts from the last covered hour. No free, token-less source of *historical + live*
   oblast-level alerts was found: `siren.pp.ua` exposes only active alerts and aggregates, the `air_alert_ua`
   Telegram channel stopped on 2026-09-07. A production version needs an alerts.in.ua token.
2. **Dataset semantics.** The volunteer dataset is a curated oblast-level record; its definition of an oblast alert
   differs from the official raion-level record. Results are about this target, not raion-level sirens.
3. **Telegram edits.** The web preview returns the current text of a post. Edits after publication could leak
   later information; the timestamp used is the original publication time. Considered low risk, not eliminated.
4. **Regex extraction** misses some places (≈8% of posts have no recognised location) and can mis-assign
   ambiguous words ("Дніпро" the river vs the city). An NER/LLM extractor could do better at a cost.
5. **Single 28-day test window.** Bootstrap CIs capture day-to-day variance, but not the variance across different
   months/regimes; a multi-month rolling backtest would be stronger.
6. **Model selection noise.** Window / hyper-parameter / Telegram-arm choices differ by ≤0.002 in validation
   log-loss — the choices themselves are close to arbitrary; the conclusions do not depend on them.
7. **Regime shifts.** Attack patterns change (Sep–Oct 2026: much more activity in central/western oblasts). The
   model adapts through 7-day features and weekly refits, but a sudden new tactic will be predicted late.
8. **Not used:** news sites, other Telegram channels (monitoring channels with more granular drone tracking),
   weather (drone activity depends on it), Russian airfield activity. Each is a plausible next feature source.
9. **Rare-alert oblasts** (Zakarpattia, Ivano-Frankivsk, Chernivtsi) have few positives; per-region AUC there is
   noisy (`reports/per_region_auc_3h.csv`).
