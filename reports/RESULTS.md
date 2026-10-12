# Test results

Test period: `2026-09-14 04:00:00+00:00` .. `2026-10-12 04:00:00+00:00` (UTC), 24 oblasts, hourly forecast origins. Data known until `2026-10-12 04:59:03+00:00`.


## Onset within 1h (positive rate 0.238)

| model | ROC-AUC | PR-AUC | Brier | log-loss | Brier skill vs climatology |
|---|---|---|---|---|---|
| Climatology (region x hour) | 0.725 | 0.461 | 0.164 | 0.513 | 0.000 |
| Seasonal naive (same hour, last 7d) | 0.710 | 0.413 | 0.169 | 0.844 | -0.029 |
| Logistic regression | 0.767 | 0.513 | 0.151 | 0.461 | 0.084 |
| **LightGBM (alerts+telegram)** | 0.788 | 0.530 | 0.146 | 0.446 | 0.111 |
| Climatology (weekly refit) | 0.729 | 0.465 | 0.163 | 0.508 | 0.006 |
| LightGBM (alerts+telegram), weekly refit | 0.791 | 0.536 | 0.145 | 0.443 | 0.117 |

## Onset within 3h (positive rate 0.501)

| model | ROC-AUC | PR-AUC | Brier | log-loss | Brier skill vs climatology |
|---|---|---|---|---|---|
| Climatology (region x hour) | 0.766 | 0.756 | 0.217 | 0.630 | 0.000 |
| Seasonal naive (same hour, last 7d) | 0.778 | 0.741 | 0.197 | 0.838 | 0.092 |
| Logistic regression | 0.805 | 0.802 | 0.181 | 0.531 | 0.167 |
| **LightGBM (alerts+telegram)** | 0.823 | 0.818 | 0.171 | 0.506 | 0.214 |
| Climatology (weekly refit) | 0.768 | 0.760 | 0.215 | 0.621 | 0.013 |
| LightGBM (alerts+telegram), weekly refit | 0.825 | 0.819 | 0.170 | 0.505 | 0.217 |

## Onset within 6h (positive rate 0.679)

| model | ROC-AUC | PR-AUC | Brier | log-loss | Brier skill vs climatology |
|---|---|---|---|---|---|
| Climatology (region x hour) | 0.827 | 0.896 | 0.184 | 0.545 | 0.000 |
| Seasonal naive (same hour, last 7d) | 0.837 | 0.886 | 0.145 | 0.718 | 0.215 |
| Logistic regression | 0.859 | 0.921 | 0.136 | 0.418 | 0.265 |
| **LightGBM (alerts+telegram)** | 0.870 | 0.929 | 0.128 | 0.398 | 0.308 |
| Climatology (weekly refit) | 0.830 | 0.898 | 0.181 | 0.535 | 0.019 |
| LightGBM (alerts+telegram), weekly refit | 0.869 | 0.928 | 0.128 | 0.398 | 0.307 |

## Stability: ROC-AUC by test week (static model / weekly-refit model / climatology)

| week starting | 1h | 3h | 6h |
|---|---|---|---|
| 2026-09-14 | 0.803 / 0.803 / 0.747 | 0.849 / 0.849 / 0.801 | 0.884 / 0.884 / 0.855 |
| 2026-09-21 | 0.790 / 0.793 / 0.746 | 0.833 / 0.835 / 0.787 | 0.889 / 0.889 / 0.852 |
| 2026-09-28 | 0.789 / 0.794 / 0.724 | 0.810 / 0.811 / 0.760 | 0.851 / 0.849 / 0.814 |
| 2026-10-05 | 0.766 / 0.773 / 0.689 | 0.796 / 0.801 / 0.719 | 0.850 / 0.851 / 0.793 |

## Telegram ablation (same training rows, period with scraped posts)

| horizon | split | arm | ROC-AUC | PR-AUC | log-loss |
|---|---|---|---|---|---|
| 1h | val | alerts_only | 0.791 | 0.507 | 0.4248 |
| 1h | test | alerts_only | 0.780 | 0.521 | 0.4513 |
| 1h | val | alerts+telegram_hourly | 0.806 | 0.527 | 0.4126 |
| 1h | test | alerts+telegram_hourly | 0.787 | 0.525 | 0.4462 |
| 1h | val | alerts+telegram | 0.806 | 0.530 | 0.4120 |
| 1h | test | alerts+telegram | 0.788 | 0.528 | 0.4457 |
| 3h | val | alerts_only | 0.839 | 0.809 | 0.4891 |
| 3h | test | alerts_only | 0.813 | 0.812 | 0.5171 |
| 3h | val | alerts+telegram_hourly | 0.849 | 0.818 | 0.4730 |
| 3h | test | alerts+telegram_hourly | 0.822 | 0.816 | 0.5080 |
| 3h | val | alerts+telegram | 0.848 | 0.819 | 0.4736 |
| 3h | test | alerts+telegram | 0.822 | 0.816 | 0.5080 |
| 6h | val | alerts_only | 0.890 | 0.925 | 0.4006 |
| 6h | test | alerts_only | 0.864 | 0.925 | 0.4093 |
| 6h | val | alerts+telegram_hourly | 0.897 | 0.929 | 0.3819 |
| 6h | test | alerts+telegram_hourly | 0.869 | 0.928 | 0.4000 |
| 6h | val | alerts+telegram | 0.897 | 0.929 | 0.3823 |
| 6h | test | alerts+telegram | 0.870 | 0.929 | 0.3979 |

## Is the improvement real? Day-block bootstrap, 95% CI of the ROC-AUC difference (test)

| horizon | comparison (final model ...) | ΔAUC | 95% CI |
|---|---|---|---|
| 1h | vs climatology | +0.0621 | [+0.0540, +0.0698] |
| 1h | vs logistic regression | +0.0204 | [+0.0144, +0.0263] |
| 1h | vs same-window alerts-only LightGBM | +0.0080 | [+0.0032, +0.0134] |
| 1h | vs telegram-hourly ablation arm (default params) | +0.0005 | [-0.0012, +0.0022] |
| 1h | weekly refit vs static | +0.0034 | [+0.0017, +0.0054] |
| 3h | vs climatology | +0.0576 | [+0.0489, +0.0658] |
| 3h | vs logistic regression | +0.0180 | [+0.0123, +0.0239] |
| 3h | vs same-window alerts-only LightGBM | +0.0099 | [+0.0061, +0.0140] |
| 3h | vs telegram-hourly ablation arm (default params) | +0.0014 | [+0.0002, +0.0026] |
| 3h | weekly refit vs static | +0.0013 | [+0.0001, +0.0025] |
| 6h | vs climatology | +0.0422 | [+0.0332, +0.0513] |
| 6h | vs logistic regression | +0.0109 | [+0.0065, +0.0154] |
| 6h | vs same-window alerts-only LightGBM | +0.0061 | [+0.0028, +0.0092] |
| 6h | vs telegram-hourly ablation arm (default params) | +0.0010 | [-0.0003, +0.0023] |
| 6h | weekly refit vs static | -0.0004 | [-0.0016, +0.0008] |

## Hyper-parameter selection (validation log-loss)

| params           |     1h |     3h |     6h |
|:-----------------|-------:|-------:|-------:|
| default          | 0.412  | 0.473  | 0.3819 |
| large_leaves     | 0.4119 | 0.4746 | 0.3838 |
| small_leaves     | 0.4122 | 0.4726 | 0.3806 |
| strong_subsample | 0.4116 | 0.4728 | 0.3811 |
