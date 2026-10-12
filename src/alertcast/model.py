"""Models, baselines and metrics."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=63, min_child_samples=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, seed=42, num_threads=4,
)


def metrics(y: np.ndarray, p: np.ndarray, base_rate_p: np.ndarray | None = None) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y)), "pos_rate": float(y.mean()),
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p),
    }
    if base_rate_p is not None:
        out["brier_skill"] = 1 - out["brier"] / brier_score_loss(y, np.clip(base_rate_p, 1e-6, 1 - 1e-6))
    return out


class Climatology:
    """P(onset) by region x local hour, estimated on the training period."""

    def fit(self, df: pd.DataFrame, y: str):
        self.table = df.groupby(["region", "hour_local"], observed=True)[y].mean()
        self.prior = df[y].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([df["region"].astype(str), df["hour_local"]])
        t = self.table.copy()
        t.index = t.index.set_levels(t.index.levels[0].astype(str), level=0)
        return t.reindex(idx).fillna(self.prior).to_numpy()


def seasonal_naive(df: pd.DataFrame, h: int) -> np.ndarray:
    """Share of the last 7 days in which an onset happened in the same window."""
    return df[f"y{h}_samehour_7d"].fillna(df[f"y{h}_samehour_7d"].mean()).to_numpy()


def fit_logreg(df: pd.DataFrame, y: str, cols: list[str]):
    num = [c for c in cols if c != "region"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["region"]),
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=300, C=0.5))
    X = df[cols].copy()
    X[num] = X[num].fillna(0)
    model.fit(X, df[y])
    return model


def predict_logreg(model, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    num = [c for c in cols if c != "region"]
    X = df[cols].copy()
    X[num] = X[num].fillna(0)
    return model.predict_proba(X)[:, 1]


def fit_lgb(train: pd.DataFrame, y: str, cols: list[str], valid: pd.DataFrame | None = None,
            num_boost_round: int = 3000) -> lgb.Booster:
    dtr = lgb.Dataset(train[cols], train[y], categorical_feature=["region"], free_raw_data=False)
    if valid is None:
        return lgb.train(LGB_PARAMS, dtr, num_boost_round=num_boost_round)
    dva = lgb.Dataset(valid[cols], valid[y], reference=dtr, categorical_feature=["region"])
    return lgb.train(LGB_PARAMS, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])
