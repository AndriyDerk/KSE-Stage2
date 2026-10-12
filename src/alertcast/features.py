"""Leakage-safe feature and target construction on an hourly forecast grid.

A forecast is issued at every hour boundary t (UTC) for every oblast.
Every feature for origin t is computed ONLY from minutes/hours strictly
before t (bins [.., t)). Targets look at [t, t+h). `tests/test_leakage.py`
verifies this by recomputing features on data truncated at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import REGION_RE, THREAT_RE, neighbours

HORIZONS = (1, 3, 6)
WINDOWS_H = (1, 3, 6, 24, 168)
MIN = pd.Timedelta(minutes=1)
HOUR = pd.Timedelta(hours=1)


def _rolling_past(x: np.ndarray, w: int) -> np.ndarray:
    """out[:, k] = sum of x[:, k-w:k]  (strictly before k)."""
    c = np.zeros((x.shape[0], x.shape[1] + 1))
    c[:, 1:] = np.cumsum(x, axis=1)
    k = np.arange(x.shape[1])
    return c[:, k] - c[:, np.maximum(k - w, 0)]


def _future_any(x: np.ndarray, h: int) -> np.ndarray:
    """out[:, k] = any(x[:, k:k+h] > 0); NaN when the window is incomplete."""
    c = np.zeros((x.shape[0], x.shape[1] + 1))
    c[:, 1:] = np.cumsum(x, axis=1)
    n = x.shape[1]
    k = np.arange(n)
    hi = np.minimum(k + h, n)
    out = (c[:, hi] - c[:, k] > 0).astype(float)
    out[:, k + h > n] = np.nan
    return out


class Grid:
    """Minute- and hour-level occupancy matrices for all regions."""

    def __init__(self, alerts: pd.DataFrame, start: str | pd.Timestamp, end: pd.Timestamp,
                 regions: list[str] | None = None):
        self.g0 = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tzinfo is None \
            else pd.Timestamp(start)
        self.end = pd.Timestamp(end).floor("h")  # last usable hour boundary
        self.regions = regions or sorted(alerts["region"].unique())
        self.ridx = {r: i for i, r in enumerate(self.regions)}
        n_hours = int((self.end - self.g0) / HOUR)
        n_min = n_hours * 60
        R = len(self.regions)
        active = np.zeros((R, n_min), dtype=bool)
        starts_h = np.zeros((R, n_hours), dtype=np.int32)
        starts_m = np.zeros((R, n_min), dtype=np.int16)
        a = alerts[(alerts["end"] > self.g0) & (alerts["start"] < self.end)]
        a = a[a["region"].isin(self.ridx)]
        s_min = ((a["start"] - self.g0) / MIN).to_numpy()
        e_min = ((a["end"] - self.g0) / MIN).to_numpy()
        ri = a["region"].map(self.ridx).to_numpy()
        for r, s, e in zip(ri, s_min, e_min):
            active[r, max(int(np.floor(s)), 0):min(int(np.ceil(e)), n_min)] = True
            if s >= 0:
                starts_h[r, int(s // 60)] += 1
                starts_m[r, int(s)] += 1
        self.n_hours, self.active, self.starts_h, self.starts_m = n_hours, active, starts_h, starts_m
        self.occ_h = active.reshape(R, n_hours, 60).mean(axis=2)
        self.hours = self.g0 + np.arange(n_hours) * HOUR

    # minute-level state at each hour boundary k (uses minute k*60-1 only)
    def boundary_state(self):
        R, M = self.active.shape
        idx = np.broadcast_to(np.arange(M, dtype=np.int32), (R, M))
        last_on = np.maximum.accumulate(np.where(self.active, idx, np.int32(-10**9)), axis=1)
        prev = np.zeros_like(self.active)
        prev[:, 1:] = self.active[:, :-1]
        run_start = np.maximum.accumulate(np.where(self.active & ~prev, idx, np.int32(-10**9)), axis=1)
        k = np.arange(self.n_hours)
        m = k * 60 - 1  # last minute before boundary
        valid = m >= 0
        mm = np.where(valid, m, 0)
        active_now = np.where(valid, self.active[:, mm], False)
        mins_since = np.where(valid, (k * 60) - last_on[:, mm], 10**6).astype(float)
        mins_since = np.minimum(mins_since, 7 * 24 * 60)
        age = np.where(active_now, (k * 60) - run_start[:, mm], 0).astype(float)
        return active_now.astype(float), mins_since, age

    def recent_starts(self, minutes: int) -> np.ndarray:
        """Alert starts in the last `minutes` before each hour boundary."""
        c = np.zeros((self.starts_m.shape[0], self.starts_m.shape[1] + 1), dtype=np.int32)
        np.cumsum(self.starts_m, axis=1, out=c[:, 1:])
        k = np.arange(self.n_hours) * 60
        return (c[:, k] - c[:, np.maximum(k - minutes, 0)]).astype(float)


def telegram_hourly(posts: pd.DataFrame, grid: Grid) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Per-hour counts of posts mentioning each region, threat-type counts,
    and a coverage mask (hours inside the scraped period)."""
    R, H = len(grid.regions), grid.n_hours
    mention = np.zeros((R, H))
    threat = {k: np.zeros(H) for k in THREAT_RE}
    total = np.zeros(H)
    p = posts.copy()
    p["ts"] = pd.to_datetime(p["published_at"], utc=True)
    p = p[(p["ts"] >= grid.g0) & (p["ts"] < grid.end)]
    hk = ((p["ts"] - grid.g0) // HOUR).astype(int).to_numpy()
    for k, text in zip(hk, p["text"].fillna("").str.lower()):
        total[k] += 1
        for r, rx in REGION_RE.items():
            if r in grid.ridx and rx.search(text):
                mention[grid.ridx[r], k] += 1
        for name, rx in THREAT_RE.items():
            if rx.search(text):
                threat[name][k] += 1
    covered = np.zeros(H, dtype=bool)
    if len(p):
        lo = int((p["ts"].min().ceil("h") - grid.g0) / HOUR)
        covered[lo + 3:] = True  # need 3h of history for the 3h windows
    return mention, threat, total, covered


def build_frame(grid: Grid, posts: pd.DataFrame | None = None, min_history_h: int = 168) -> pd.DataFrame:
    """Long frame: one row per (origin hour, region) with features + targets."""
    R, H = len(grid.regions), grid.n_hours
    active_now, mins_since, age = grid.boundary_state()
    nb = neighbours()
    nb_mat = np.zeros((R, R))
    for r, i in grid.ridx.items():
        for n in nb.get(r, ()):
            if n in grid.ridx:
                nb_mat[i, grid.ridx[n]] = 1
    nb_cnt = np.maximum(nb_mat.sum(1, keepdims=True), 1)

    cols: dict[str, np.ndarray] = {
        "active_now": active_now,
        "mins_since_alert": mins_since,
        "current_alert_age": age,
        "nat_active_now": np.broadcast_to(active_now.sum(0), (R, H)),
        "nb_active_now": nb_mat @ active_now / nb_cnt,
    }
    for w in WINDOWS_H:
        occ = _rolling_past(grid.occ_h, w) / w
        st = _rolling_past(grid.starts_h, w)
        cols[f"occ_{w}h"] = occ
        cols[f"starts_{w}h"] = st
        if w <= 24:
            cols[f"nat_occ_{w}h"] = np.broadcast_to(occ.mean(0), (R, H))
            cols[f"nat_starts_{w}h"] = np.broadcast_to(st.sum(0), (R, H))
            cols[f"nb_occ_{w}h"] = nb_mat @ occ / nb_cnt
            cols[f"nb_starts_{w}h"] = nb_mat @ st
    # sub-hour dynamics: alert waves move between neighbouring oblasts in minutes
    for w in (15, 30):
        st = grid.recent_starts(w)
        cols[f"starts_{w}m"] = st
        cols[f"nb_starts_{w}m"] = nb_mat @ st
        cols[f"nat_starts_{w}m"] = np.broadcast_to(st.sum(0), (R, H))
    # change in national activity: last hour vs the 3 hours before it
    cols["nat_starts_trend"] = cols["nat_starts_1h"] - (cols["nat_starts_3h"] - cols["nat_starts_1h"]) / 2

    targets = {h: _future_any(grid.starts_h, h) for h in HORIZONS}
    # Seasonal lags of the target: windows [k-24d, k-24d+h) end before k for h<=24.
    for h in HORIZONS:
        y = targets[h]
        lag = lambda d: np.concatenate([np.full((R, 24 * d), np.nan), y[:, :-24 * d]], axis=1)  # noqa: E731
        cols[f"y{h}_lag24"] = lag(1)
        cols[f"y{h}_samehour_7d"] = np.nanmean(np.stack([lag(d) for d in range(1, 8)]), axis=0)

    if posts is not None:
        mention, threat, total, covered = telegram_hourly(posts, grid)
        for w in (1, 3):
            m = _rolling_past(mention, w)
            cols[f"tg_mention_{w}h"] = m
            cols[f"tg_nb_mention_{w}h"] = nb_mat @ m
            cols[f"tg_posts_{w}h"] = np.broadcast_to(_rolling_past(total[None], w)[0], (R, H))
            for name, v in threat.items():
                cols[f"tg_{name}_{w}h"] = np.broadcast_to(_rolling_past(v[None], w)[0], (R, H))
        cols["tg_covered"] = np.broadcast_to(covered, (R, H)).astype(float)

    df = pd.DataFrame({k: np.asarray(v, dtype=float).ravel() for k, v in cols.items()})
    for h in HORIZONS:
        df[f"y_{h}h"] = targets[h].ravel()
    df["region"] = np.repeat(grid.regions, H)
    df["t"] = np.tile(grid.hours, R)
    local = df["t"].dt.tz_convert("Europe/Kyiv")
    df["hour_local"] = local.dt.hour
    df["dow"] = local.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_local"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_local"] / 24)
    df = df[df["t"] >= grid.g0 + min_history_h * HOUR]
    df["region"] = pd.Categorical(df["region"], categories=grid.regions)
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame, telegram: bool) -> list[str]:
    drop = {"t", "tg_covered"} | {c for c in df.columns if c.startswith("y_")}
    cols = [c for c in df.columns if c not in drop]
    if not telegram:
        cols = [c for c in cols if not c.startswith("tg_")]
    return cols
