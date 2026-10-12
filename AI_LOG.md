# AI process log (summary)

The primary AI tool was **Claude Code** (agentic mode: it wrote and ran the code, read the data and the
outputs, and iterated). The full, unedited conversation export is submitted separately via the form; this file
is a condensed index of the decisions and course corrections that happened in it.

| # | step | what was found / challenged | decision |
|---|---|---|---|
| 1 | Scope | Several possible targets (alert duration, active/inactive, onset). | Onset of a new alert within 1/3/6 h per oblast — "is active" is trivial via persistence. |
| 2 | Data choice | `official_data_en.csv` looked like the obvious source, but its last record is **2026-09-07**, and in 2026 alerts are recorded at raion level. `volunteer_data_en.csv` is oblast-level and updated to **2026-10-12**. Spot check (Kharkivska, Aug 2026): the two files differ in what they count. | Use the volunteer file (latest data, as the task requires), document the difference. |
| 3 | Leakage design | Rolling features on hourly bins can silently include the current hour; running alerts have unknown end times. | Minute grid, features strictly from `[.., t)`, running alerts clipped at `t`; seasonal target lags only for windows that end before `t`. Wrote an automated test that recomputes features on data truncated at `t` (synthetic + real data). |
| 4 | Splits | Random CV would leak through autocorrelation; adjacent splits leak through the target window. | Train / 28-day validation / last 28-day test, with an `h`-hour embargo. Hyper-choices only on validation; test scored once. |
| 5 | Baselines | A strong model is meaningless without them. | Climatology (region × hour), seasonal naive (same hour, last 7 days), logistic regression. |
| 6 | First model | LightGBM beats all baselines, but feature importance shows it mostly learns *base rate* (`starts_168h`, `region`). | Hypothesis: hourly windows too coarse to catch drone waves moving between oblasts. |
| 7 | Sub-hour features | Added 15/30-min start counts (own, neighbours, national). Result: AUC 0.7797 → 0.7798. | Hypothesis rejected (oblast alerts in this dataset are declared near-simultaneously; `nat_occ_1h` already carries it). Kept as a documented negative result. |
| 8 | Training window | Is old data (2024) harmful after regime changes? | Compared 4 windows on validation: differences are tiny (log-loss 0.4241–0.4275); the best one is chosen per horizon. |
| 9 | Telegram | `t.me/s/kpszsu` is scrapable without login (24.5k posts since May 2026). | Region-mention and threat-type counts in the last 1/3 h; only posts published before `t`. |
| 10 | Regex bugs (own error) | Inspecting matches on real posts showed: "балістичного" not matched by stem "балістик" (86 hits on 24k posts!), "Миколаєва"/"Харкова"/"Запорізьку" missed due to Ukrainian sound alternations, "Одещина"/"Черкащина" missed, and many posts name towns, not oblasts (Кривий Ріг, Бориспіль, Дарниця). "Відбій загрози балістики" was counted as a threat. | Fixed stems, added major towns and districts, added separate `all_clear` / `nationwide` features. Posts without a recognised place: 4,569 → 2,070. |
| 11 | Telegram ablation | A fair comparison must use the same rows and window for both arms. | Telegram improves validation log-loss on all horizons and test AUC by +0.005…+0.010 → chosen on validation. |
| 12 | Model aging | Weekly breakdown of the test: AUC drops from 0.80 to 0.76 over 4 weeks. | Added walk-forward evaluation (refit every 7 days on data before that week) — recovers part of the drop; recommended deployment mode. |
| 13 | A wrong impression (own error) | The Kyiv timeline plot looked over-confident. Checked numerically before "fixing" anything: mean predicted 0.527 vs observed 0.507 — calibrated. The real story was climatology: Kyivska oblast 85% observed vs 46% climatology (regime shift in Sep–Oct 2026). | No change to the model; the finding went into the README. |
