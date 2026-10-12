"""Live forecast: P(new air-raid alert starts within 1/3/6 h) for every oblast,
issued at the latest hour boundary covered by the data.

Usage: python scripts/predict.py [--download] [--scrape]
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alertcast.data import download, load_alerts  # noqa: E402
from alertcast.features import HORIZONS, Grid, build_frame  # noqa: E402

TG = ROOT / "data/raw/telegram_kpszsu.csv"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch the latest alert dataset first")
    ap.add_argument("--scrape", action="store_true", help="fetch the latest Telegram posts first")
    a = ap.parse_args()
    if a.download:
        download(ROOT / "data/raw")
    if a.scrape:
        from alertcast.telegram import scrape_channel
        scrape_channel("kpszsu", "2025-12-01", TG)
    meta = json.loads((ROOT / "models/meta.json").read_text())
    alerts = load_alerts(ROOT / "data/raw/volunteer_data_en.csv")
    known = max(alerts["start"].max(), alerts["end"].max())
    origin = known.floor("h")
    posts = pd.read_csv(TG) if TG.exists() else None
    if posts is not None:  # same rule as in training: only posts published before the origin
        posts = posts[pd.to_datetime(posts["published_at"], utc=True) < origin]
    grid = Grid(alerts, origin - pd.Timedelta(days=21), origin + pd.Timedelta(hours=1))
    df = build_frame(grid, posts, min_history_h=0)
    now = df[df["t"] == origin].copy()
    for h in HORIZONS:
        m = meta[f"final_{h}h"]
        if any(c.startswith("tg_") for c in m["features"]) and posts is None:
            sys.exit("model uses Telegram features: run with --scrape")
        now[f"P_{h}h"] = lgb.Booster(model_file=str(ROOT / f"models/lgb_{h}h.txt")).predict(now[m["features"]])
    # an onset within 1h implies an onset within 3h and 6h: enforce P_1h <= P_3h <= P_6h
    # (independently trained models violate it for <1% of rows, by up to 0.2)
    for a, b in zip(HORIZONS, HORIZONS[1:]):
        now[f"P_{b}h"] = now[[f"P_{a}h", f"P_{b}h"]].max(axis=1)
    out = now[["region", "active_now"] + [f"P_{h}h" for h in HORIZONS]].sort_values("P_3h", ascending=False)
    out["region"] = out["region"].astype(str)
    print(f"Forecast issued at {origin} UTC ({origin.tz_convert('Europe/Kyiv'):%Y-%m-%d %H:%M} Kyiv), "
          f"data known until {known}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    out.assign(origin_utc=origin).to_csv(ROOT / "reports/latest_forecast.csv", index=False)
