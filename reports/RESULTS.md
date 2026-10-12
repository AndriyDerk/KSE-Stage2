# Test results

Test period: `2026-09-14 04:00:00+00:00` .. `2026-10-12 04:00:00+00:00` (UTC), 24 oblasts, hourly forecast origins. Data known until `2026-10-12 04:59:03+00:00`.


## Onset within 1h (positive rate 0.238)

| model | ROC-AUC | PR-AUC | Brier | log-loss | Brier skill vs climatology |
|---|---|---|---|---|---|
| Climatology (region x hour) | 0.733 | 0.471 | 0.160 | 0.505 | 0.000 |
| Seasonal naive (same hour, last 7d) | 0.710 | 0.413 | 0.169 | 0.844 | -0.055 |
| Logistic regression | 0.767 | 0.513 | 0.150 | 0.460 | 0.063 |
| **LightGBM (alerts+telegram)** | 0.786 | 0.523 | 0.147 | 0.447 | 0.084 |
| Climatology (weekly refit) | 0.738 | 0.479 | 0.159 | 0.498 | 0.008 |
| LightGBM (alerts+telegram), weekly refit | 0.790 | 0.530 | 0.146 | 0.444 | 0.092 |

## Onset within 3h (positive rate 0.501)

| model | ROC-AUC | PR-AUC | Brier | log-loss | Brier skill vs climatology |
|---|---|---|---|---|---|
| Climatology (region x hour) | 0.775 | 0.767 | 0.207 | 0.605 | 0.000 |
| Seasonal naive (same hour, last 7d) | 0.778 | 0.741 | 0.197 | 0.838 | 0.049 |
| Logistic regression | 0.805 | 0.802 | 0.180 | 0.529 | 0.130 |
| **LightGBM (alerts+telegram)** | 0.820 | 0.816 | 0.172 | 0.509 | 0.170 |
| Climatology (weekly refit) | 0.778 | 0.772 | 0.203 | 0.592 | 0.019 |
| LightGBM (alerts+telegram), weekly refit | 0.823 | 0.818 | 0.171 | 0.506 | 0.177 |

## Onset within 6h (positive rate 0.679)

| model | ROC-AUC | PR-AUC | Brier | log-loss | Brier skill vs climatology |
|---|---|---|---|---|---|
| Climatology (region x hour) | 0.834 | 0.901 | 0.173 | 0.517 | 0.000 |
| Seasonal naive (same hour, last 7d) | 0.837 | 0.886 | 0.145 | 0.718 | 0.162 |
| Logistic regression | 0.859 | 0.921 | 0.135 | 0.416 | 0.221 |
| **LightGBM (alerts+telegram)** | 0.866 | 0.926 | 0.130 | 0.405 | 0.250 |
| Climatology (weekly refit) | 0.837 | 0.904 | 0.167 | 0.502 | 0.031 |
| LightGBM (alerts+telegram), weekly refit | 0.867 | 0.927 | 0.129 | 0.402 | 0.256 |

## Stability: ROC-AUC by test week (static model / weekly-refit model / climatology)

| week starting | 1h | 3h | 6h |
|---|---|---|---|
| 2026-09-14 | 0.803 / 0.803 / 0.755 | 0.846 / 0.846 / 0.810 | 0.881 / 0.881 / 0.862 |
| 2026-09-21 | 0.788 / 0.790 / 0.754 | 0.831 / 0.833 / 0.797 | 0.885 / 0.885 / 0.859 |
| 2026-09-28 | 0.790 / 0.794 / 0.731 | 0.809 / 0.811 / 0.765 | 0.850 / 0.850 / 0.817 |
| 2026-10-05 | 0.760 / 0.771 / 0.696 | 0.791 / 0.799 / 0.731 | 0.845 / 0.848 / 0.803 |

## Telegram ablation (same training rows, period with scraped posts)

| horizon | split | arm | ROC-AUC | PR-AUC | log-loss |
|---|---|---|---|---|---|
| 1h | val | alerts_only | 0.788 | 0.506 | 0.4271 |
| 1h | test | alerts_only | 0.776 | 0.520 | 0.4532 |
| 1h | val | alerts+telegram | 0.803 | 0.525 | 0.4152 |
| 1h | test | alerts+telegram | 0.786 | 0.523 | 0.4474 |
| 3h | val | alerts_only | 0.836 | 0.804 | 0.4954 |
| 3h | test | alerts_only | 0.811 | 0.809 | 0.5200 |
| 3h | val | alerts+telegram | 0.847 | 0.815 | 0.4773 |
| 3h | test | alerts+telegram | 0.820 | 0.816 | 0.5094 |
| 6h | val | alerts_only | 0.886 | 0.922 | 0.4121 |
| 6h | test | alerts_only | 0.861 | 0.924 | 0.4135 |
| 6h | val | alerts+telegram | 0.893 | 0.926 | 0.3925 |
| 6h | test | alerts+telegram | 0.866 | 0.926 | 0.4052 |
