"""Walk-forward backtest and scoring.

For each test year Y the models are refitted on every played test from TRAIN_FROM up to
31 December of Y-1 and then predict all of year Y. No model ever sees a result from its test year,
and the features themselves are pre-match (see features.py).
"""
import numpy as np
import pandas as pd

from . import teams
from . import model as md
from .model import Ensemble, all_models

TRAIN_FROM = "1995-01-01"  # professional era
FIRST_TEST_YEAR = 2008


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def accuracy(p, y):
    """Share of decisive matches where the favourite won."""
    p, y = np.asarray(p), np.asarray(y)
    decisive = y != 0.5
    return float(np.mean((p[decisive] > 0.5) == (y[decisive] == 1)))


def tier1_mask(df):
    return df.home.isin(teams.TIER1) & df.away.isin(teams.TIER1)


def walk_forward(feat, first_year=FIRST_TEST_YEAR, last_year=None, models=None, train_from=TRAIN_FROM):
    """Return the played test-period rows with one prediction column per model."""
    played = feat[feat.played].copy()
    last_year = last_year or played.date.dt.year.max()
    models = models or all_models()
    out = []
    for year in range(first_year, last_year + 1):
        train = played[(played.date >= train_from) & (played.date < f"{year}-01-01")]
        test = played[played.date.dt.year == year].copy()
        if test.empty:
            continue
        parts = []
        for m in models:
            if isinstance(m, Ensemble):
                continue  # the ensemble is the average of its parts, filled in below
            test[m.name] = m.fit(train).predict(test)
            parts.append(m.name)
        ens = Ensemble()
        test[ens.name] = test[[p.name for p in ens.parts]].mean(axis=1)
        mm = next(m for m in models if m.name.startswith("Margin"))
        test["pred_margin"] = mm.margin.predict(test[md.FEATURES])  # active inputs (ablation-aware)
        out.append(test)
    return pd.concat(out, ignore_index=True)


def model_columns(preds):
    return [m.name for m in all_models() if m.name in preds.columns]


def score_table(preds, mask=None):
    df = preds if mask is None else preds[mask]
    rows = []
    for c in model_columns(df):
        rows.append({"model": c, "matches": len(df), "log_loss": log_loss(df[c], df.result),
                     "brier": brier(df[c], df.result), "accuracy": accuracy(df[c], df.result)})
    return pd.DataFrame(rows).set_index("model")


def calibration(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"p": p, "y": y, "b": idx})
    g = df.groupby("b").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    return g[g.n > 0]
