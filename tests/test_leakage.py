"""Leakage tests: features for origin t must not change when every piece of
information from >= t is removed from the input."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertcast.data import load_alerts, truncate_at  # noqa: E402
from alertcast.features import Grid, build_frame, feature_columns  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data/raw/volunteer_data_en.csv"
TG = Path(__file__).resolve().parents[1] / "data/raw/telegram_kpszsu.csv"


def _synthetic():
    rng = np.random.default_rng(0)
    regions = ["Kyiv City", "Kyivska oblast", "Lvivska oblast", "Kharkivska oblast"]
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for r in regions:
        t = t0
        while t < t0 + pd.Timedelta(days=20):
            t += pd.Timedelta(minutes=float(rng.exponential(300)))
            rows.append((r, t, t + pd.Timedelta(minutes=float(rng.exponential(60)) + 1)))
    posts = pd.DataFrame({
        "post_id": range(200),
        "published_at": [str(t0 + pd.Timedelta(minutes=int(m))) for m in sorted(rng.integers(0, 20 * 1440, 200))],
        "text": rng.choice(["БпЛА на Харківщину", "Балістика на Київ", "Львів: ракети", "чисто"], 200),
    })
    return pd.DataFrame(rows, columns=["region", "start", "end"]), posts


def _check(alerts, posts, start, cut, end):
    full = build_frame(Grid(alerts, start, end), posts)
    trunc_posts = None if posts is None else posts[pd.to_datetime(posts["published_at"], utc=True) < cut]
    part = build_frame(Grid(truncate_at(alerts, cut), start, end, regions=sorted(alerts["region"].unique())),
                       trunc_posts)
    cols = [c for c in feature_columns(full, telegram=posts is not None) if c != "region"]
    a = full[full["t"] <= cut].set_index(["t", "region"])[cols].sort_index()
    b = part[part["t"] <= cut].set_index(["t", "region"])[cols].sort_index()
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("cut_h", [200, 333, 401])
def test_no_leakage_synthetic(cut_h):
    alerts, posts = _synthetic()
    start = pd.Timestamp("2026-01-01", tz="UTC")
    _check(alerts, posts, start, start + pd.Timedelta(hours=cut_h), start + pd.Timedelta(days=21))


@pytest.mark.skipif(not RAW.exists(), reason="raw data not downloaded")
def test_no_leakage_real_data():
    alerts = load_alerts(RAW)
    posts = pd.read_csv(TG) if TG.exists() else None
    cut = pd.Timestamp("2026-09-20 13:00", tz="UTC")
    _check(alerts, posts, pd.Timestamp("2026-08-01", tz="UTC"), cut, pd.Timestamp("2026-10-01", tz="UTC"))


def test_targets_look_forward_only():
    alerts, _ = _synthetic()
    start = pd.Timestamp("2026-01-01", tz="UTC")
    df = build_frame(Grid(alerts, start, start + pd.Timedelta(days=21)))
    row = df.iloc[len(df) // 2]
    t, r = row["t"], row["region"]
    a = alerts[alerts["region"] == r]
    expect = ((a["start"] >= t) & (a["start"] < t + pd.Timedelta(hours=3))).any()
    assert bool(row["y_3h"]) == bool(expect)
