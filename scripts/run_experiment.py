"""End-to-end experiment: features -> time-based splits -> baselines/models ->
validation-based selection -> single evaluation on the latest 28 days.

Usage:  python scripts/run_experiment.py [--download]
Outputs: reports/*.csv, reports/*.md, reports/figures/*.png, models/*.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alertcast import model as M  # noqa: E402
from alertcast.data import download, load_alerts  # noqa: E402
from alertcast.features import HORIZONS, Grid, build_frame, feature_columns  # noqa: E402

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
MODELS = ROOT / "models"
TG_PATH = ROOT / "data/raw/telegram_kpszsu.csv"
GRID_START = "2024-06-01"
TEST_DAYS = VAL_DAYS = 28
WINDOWS = {"since_2024-06": "2024-06-08", "since_2025-06": "2025-06-01",
           "since_2026-01": "2026-01-01", "since_2026-05": "2026-05-08"}
H = pd.Timedelta(hours=1)
D = pd.Timedelta(days=1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def split_bounds(end: pd.Timestamp):
    test_start = end - TEST_DAYS * D
    val_start = test_start - VAL_DAYS * D
    return val_start, test_start


def subset(df, lo, hi, h):
    """Origins in [lo, hi - h]: targets [t, t+h) stay inside [lo, hi) (embargo)."""
    m = (df["t"] >= lo) & (df["t"] + h * H <= hi) & df[f"y_{h}h"].notna()
    return df[m]


def main(dl: bool):
    REPORTS.mkdir(exist_ok=True), FIG.mkdir(exist_ok=True), MODELS.mkdir(exist_ok=True)
    if dl or not (ROOT / "data/raw/volunteer_data_en.csv").exists():
        download(ROOT / "data/raw")
    alerts = load_alerts(ROOT / "data/raw/volunteer_data_en.csv")
    known_until = max(alerts["start"].max(), alerts["end"].max())
    posts = pd.read_csv(TG_PATH) if TG_PATH.exists() else None
    grid = Grid(alerts, GRID_START, known_until)
    df = build_frame(grid, posts)
    end = grid.end
    val_start, test_start = split_bounds(end)
    log(f"data known until {known_until}; grid end {end}; rows {len(df):,}")
    log(f"val  [{val_start} .. {test_start}),  test [{test_start} .. {end})")
    base_cols = feature_columns(df, telegram=False)
    tg_cols = feature_columns(df, telegram=True) if posts is not None else None
    if posts is not None:
        tg_start = df.loc[df["tg_covered"] > 0, "t"].min()
        log(f"telegram posts: {len(posts):,}, features available from {tg_start}")

    meta = {"known_until": str(known_until), "grid_end": str(end), "val_start": str(val_start),
            "test_start": str(test_start), "n_regions": len(grid.regions)}
    sel_rows, test_rows, tg_rows, imp_rows = [], [], [], []
    test_preds = []

    for h in HORIZONS:
        y = f"y_{h}h"
        val = subset(df, val_start, test_start, h)
        test = subset(df, test_start, end + H, h)

        # 1) training-window selection on validation (LightGBM, alert features)
        best = None
        for name, w0 in WINDOWS.items():
            tr = subset(df, pd.Timestamp(w0, tz="UTC"), val_start, h)
            clim = M.Climatology().fit(tr, y)
            bst = M.fit_lgb(tr, y, base_cols, val)
            p = bst.predict(val[base_cols], num_iteration=bst.best_iteration)
            r = M.metrics(val[y].to_numpy(), p, clim.predict(val))
            r.update(horizon=h, window=name, best_iter=bst.best_iteration, n_train=len(tr))
            sel_rows.append(r)
            log(f"h={h} window={name:14s} val AUC={r['roc_auc']:.4f} LL={r['log_loss']:.4f} it={bst.best_iteration}")
            if best is None or r["log_loss"] < best[1]["log_loss"]:
                best = (name, r)
        w_name, w_res = best

        # 2) Telegram ablation on the period where posts exist (same rows for both arms)
        use_tg = False
        if tg_cols is not None:
            tr = subset(df[df["tg_covered"] > 0], tg_start, val_start, h)
            res = {}
            for arm, cols in (("alerts_only", base_cols), ("alerts+telegram", tg_cols)):
                bst = M.fit_lgb(tr, y, cols, val)
                pv = bst.predict(val[cols], num_iteration=bst.best_iteration)
                trv = subset(df[df["tg_covered"] > 0], tg_start, test_start, h)
                bst2 = M.fit_lgb(trv, y, cols, num_boost_round=max(50, int(bst.best_iteration * 1.1)))
                pt = bst2.predict(test[cols])
                clim = M.Climatology().fit(trv, y)
                for split, yy, pp, base in (("val", val[y], pv, M.Climatology().fit(tr, y).predict(val)),
                                            ("test", test[y], pt, clim.predict(test))):
                    r = M.metrics(yy.to_numpy(), pp, base)
                    r.update(horizon=h, arm=arm, split=split, n_train=len(tr))
                    tg_rows.append(r)
                res[arm] = tg_rows[-2]["log_loss"]  # validation log-loss
                log(f"h={h} tg-ablation {arm:16s} val LL={tg_rows[-2]['log_loss']:.4f} "
                    f"AUC={tg_rows[-2]['roc_auc']:.4f} | test AUC={tg_rows[-1]['roc_auc']:.4f}")
            use_tg = res["alerts+telegram"] < res["alerts_only"]

        # 3) final model chosen on validation, refit on train+val, evaluated ONCE on test
        if use_tg:
            final_cols, w0, final_name = tg_cols, tg_start, "LightGBM (alerts+telegram)"
            tr_fit = subset(df[df["tg_covered"] > 0], w0, val_start, h)
            n_iter = M.fit_lgb(tr_fit, y, final_cols, val).best_iteration
        else:
            final_cols, w0, final_name = base_cols, pd.Timestamp(WINDOWS[w_name], tz="UTC"), "LightGBM (alerts)"
            n_iter = w_res["best_iter"]
        trv = subset(df, w0, test_start, h)
        if use_tg:
            trv = trv[trv["tg_covered"] > 0]
        final = M.fit_lgb(trv, y, final_cols, num_boost_round=max(50, int(n_iter * 1.1)))
        final.save_model(str(MODELS / f"lgb_{h}h.txt"))
        clim = M.Climatology().fit(trv, y)
        logreg = M.fit_logreg(trv, y, base_cols)
        base_p = clim.predict(test)
        preds = {
            "Climatology (region x hour)": base_p,
            "Seasonal naive (same hour, last 7d)": M.seasonal_naive(test, h),
            "Logistic regression": M.predict_logreg(logreg, test, base_cols),
            final_name: final.predict(test[final_cols]),
        }
        # walk-forward: refit every 7 days on everything known before that week
        wf, wf_clim = np.zeros(len(test)), np.zeros(len(test))
        for ws in pd.date_range(test_start, end, freq="7D"):
            m = ((test["t"] >= ws) & (test["t"] < ws + 7 * D)).to_numpy()
            if not m.any():
                continue
            trw = subset(df, w0, ws, h)
            if use_tg:
                trw = trw[trw["tg_covered"] > 0]
            wf[m] = M.fit_lgb(trw, y, final_cols, num_boost_round=max(50, int(n_iter * 1.1))).predict(
                test.loc[m, final_cols])
            wf_clim[m] = M.Climatology().fit(trw, y).predict(test[m])
        preds["Climatology (weekly refit)"] = wf_clim
        preds[final_name + ", weekly refit"] = wf
        for name, p in preds.items():
            r = M.metrics(test[y].to_numpy(), p, base_p)
            r.update(horizon=h, model=name, final=name == final_name)
            test_rows.append(r)
            log(f"h={h} TEST {name:40s} AUC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} "
                f"Brier={r['brier']:.4f} BSS={r['brier_skill']:.3f}")
        imp = pd.Series(final.feature_importance("gain"), index=final_cols)
        imp_rows.append((imp / imp.sum()).rename(f"{h}h"))
        tp = test[["t", "region", y]].rename(columns={y: "y"}).assign(
            horizon=h, p=preds[final_name], p_wf=wf, p_clim=base_p, p_clim_wf=wf_clim)
        test_preds.append(tp)
        meta[f"final_{h}h"] = {"model": final_name, "train_from": str(w0), "n_iter": int(n_iter * 1.1),
                               "features": final_cols}

    pd.DataFrame(sel_rows).to_csv(REPORTS / "val_window_selection.csv", index=False)
    pd.DataFrame(test_rows).to_csv(REPORTS / "test_metrics.csv", index=False)
    if tg_rows:
        pd.DataFrame(tg_rows).to_csv(REPORTS / "telegram_ablation.csv", index=False)
    pd.concat(imp_rows, axis=1).sort_values("1h", ascending=False).to_csv(REPORTS / "feature_importance.csv")
    preds_df = pd.concat(test_preds)
    preds_df["region"] = preds_df["region"].astype(str)
    preds_df.to_parquet(REPORTS / "test_predictions.parquet") if _has_parquet() else \
        preds_df.to_csv(REPORTS / "test_predictions.csv.gz", index=False)
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (REPORTS / "run_meta.json").write_text(json.dumps({k: v for k, v in meta.items()
                                                       if not k.startswith("final")} |
                                                      {k: {kk: vv for kk, vv in v.items() if kk != "features"}
                                                       for k, v in meta.items() if k.startswith("final")},
                                                      indent=2))
    log("done")


def _has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="re-download the latest alert dataset")
    main(ap.parse_args().download)
