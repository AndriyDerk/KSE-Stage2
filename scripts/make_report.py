"""Figures + markdown summary from the outputs of run_experiment.py."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, FIG = ROOT / "reports", ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})
C_MODEL, C_BASE = "#1f5fa8", "#9a9a9a"

p = R / "test_predictions.parquet"
preds = pd.read_parquet(p) if p.exists() else pd.read_csv(R / "test_predictions.csv.gz", parse_dates=["t"])
metrics = pd.read_csv(R / "test_metrics.csv")
meta = json.loads((R / "run_meta.json").read_text())

# 1. calibration
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, (h, g) in zip(axes, preds.groupby("horizon")):
    for col, c, lab in (("p", C_MODEL, "final model"), ("p_clim", C_BASE, "climatology")):
        b = pd.cut(g[col], np.linspace(0, 1, 11), include_lowest=True)
        cal = g.groupby(b, observed=True).agg(pred=(col, "mean"), obs=("y", "mean"), n=("y", "size"))
        cal = cal[cal["n"] >= 30]
        ax.plot(cal["pred"], cal["obs"], "o-", color=c, label=lab, ms=4)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"Onset within {h}h"), ax.set_xlabel("predicted probability")
axes[0].set_ylabel("observed frequency"), axes[0].legend(frameon=False)
fig.suptitle("Calibration on the test period (last 28 days)")
fig.tight_layout(), fig.savefig(FIG / "calibration.png"), plt.close(fig)

# 2. per-region AUC (3h)
g = preds[preds["horizon"] == 3]
rows = []
for r, x in g.groupby("region"):
    if x["y"].nunique() == 2:
        rows.append((r.replace(" oblast", ""), roc_auc_score(x["y"], x["p"]), roc_auc_score(x["y"], x["p_clim"]),
                     x["y"].mean()))
reg = pd.DataFrame(rows, columns=["region", "model", "climatology", "pos_rate"]).sort_values("model")
reg.to_csv(R / "per_region_auc_3h.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 7))
yy = np.arange(len(reg))
ax.barh(yy, reg["model"], color=C_MODEL, height=0.6, label="final model")
ax.scatter(reg["climatology"], yy, color="k", marker="|", s=120, label="climatology", zorder=3)
ax.set_yticks(yy, reg["region"]), ax.set_xlim(0.5, 1), ax.set_xlabel("ROC-AUC, onset within 3h (test)")
ax.legend(frameon=False, loc="lower right"), fig.tight_layout(), fig.savefig(FIG / "per_region_auc_3h.png")
plt.close(fig)

# 3. example timeline
for region in ("Kyiv City", "Kharkivska oblast"):
    x = preds[(preds["horizon"] == 3) & (preds["region"] == region)].sort_values("t")
    x = x[x["t"] >= x["t"].max() - pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(x["t"], x["p"], color=C_MODEL, lw=1.2, label="P(onset within 3h)")
    ax.plot(x["t"], x["p_clim"], color=C_BASE, lw=1, label="climatology")
    ax.fill_between(x["t"], 0, x["y"], step="post", color="#d62728", alpha=0.15, label="actual: onset within 3h")
    ax.set_ylim(0, 1), ax.legend(frameon=False, ncol=3, loc="upper left"), ax.set_title(f"{region}: last 7 test days")
    fig.tight_layout(), fig.savefig(FIG / f"timeline_{region.split()[0].lower()}.png"), plt.close(fig)

# 4. feature importance
imp = pd.read_csv(R / "feature_importance.csv", index_col=0)
top = imp["3h"].sort_values().tail(15)
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(top.index, top.values, color=C_MODEL), ax.set_xlabel("share of total gain (3h model)")
fig.tight_layout(), fig.savefig(FIG / "feature_importance_3h.png"), plt.close(fig)

# markdown summary
fmt = metrics.copy()
for c in ("roc_auc", "pr_auc", "brier", "log_loss", "brier_skill", "pos_rate"):
    fmt[c] = fmt[c].map(lambda v: f"{v:.3f}")
lines = [f"# Test results\n", f"Test period: `{meta['test_start']}` .. `{meta['grid_end']}` (UTC), "
         f"{meta['n_regions']} oblasts, hourly forecast origins. Data known until `{meta['known_until']}`.\n"]
for h, g in fmt.groupby("horizon"):
    lines.append(f"\n## Onset within {h}h (positive rate {g['pos_rate'].iloc[0]})\n")
    lines.append("| model | ROC-AUC | PR-AUC | Brier | log-loss | Brier skill vs climatology |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        name = f"**{r['model']}**" if r["final"] else r["model"]
        lines.append(f"| {name} | {r['roc_auc']} | {r['pr_auc']} | {r['brier']} | {r['log_loss']} | {r['brier_skill']} |")
lines.append("\n## Stability: ROC-AUC by test week (static model / weekly-refit model / climatology)\n")
lines.append("| week starting | 1h | 3h | 6h |")
lines.append("|---|---|---|---|")
preds["week"] = (preds["t"] - preds["t"].min()).dt.days // 7
for w, gw in preds.groupby("week"):
    cells = []
    for h, gh in gw.groupby("horizon"):
        cells.append(" / ".join(f"{roc_auc_score(gh['y'], gh[c]):.3f}" for c in ("p", "p_wf", "p_clim")))
    lines.append(f"| {gw['t'].min():%Y-%m-%d} | " + " | ".join(cells) + " |")
ab = R / "telegram_ablation.csv"
if ab.exists():
    a = pd.read_csv(ab)
    lines.append("\n## Telegram ablation (same training rows, period with scraped posts)\n")
    lines.append("| horizon | split | arm | ROC-AUC | PR-AUC | log-loss |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in a.iterrows():
        lines.append(f"| {r['horizon']}h | {r['split']} | {r['arm']} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | "
                     f"{r['log_loss']:.4f} |")
(R / "RESULTS.md").write_text("\n".join(lines) + "\n")
print((R / "RESULTS.md").read_text())
