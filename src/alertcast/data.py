"""Loading the air-raid alert history.

Primary source: the volunteer oblast-level dataset maintained in
github.com/Vadimkin/ukrainian-air-raid-sirens-dataset (updated daily).
The "official" file in the same repo stopped updating on 2026-09-07 and, in
2026, records most alerts at raion/hromada level, so it is not used for the
target (see README, "Data").
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/"
VOLUNTEER = "volunteer_data_en.csv"

# Luhanska oblast is almost fully occupied: 3 records in 4.5 years.
EXCLUDED = {"Luhanska oblast"}


def download(dest_dir: Path = Path("data/raw"), name: str = VOLUNTEER) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / name
    r = requests.get(BASE + name, timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def load_alerts(path: Path = Path("data/raw") / VOLUNTEER) -> pd.DataFrame:
    """Return alerts as (region, start, end) with UTC timestamps, cleaned."""
    df = pd.read_csv(path)
    df = df.rename(columns={"started_at": "start", "finished_at": "end"})
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)
    df = df[~df["region"].isin(EXCLUDED)]
    df = df[df["end"] > df["start"]]  # 6 zero-length records
    return df[["region", "start", "end"]].sort_values("start").reset_index(drop=True)


def truncate_at(alerts: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """What was knowable at time t: alerts started before t; ongoing ones are
    clipped at t (their end is not yet known)."""
    a = alerts[alerts["start"] < t].copy()
    a["end"] = a["end"].clip(upper=t)
    return a
