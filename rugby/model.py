"""Match models on the 20 features, plus the baselines they are judged against.

Draws (about 2% of tests) count as half a win: each draw enters training twice, once as a
win and once as a loss, at weight 0.5.

Training rows are weighted towards recent matches (exponential decay, time constant tau years)
and towards tier-1 v tier-1 tests, which are otherwise outnumbered by tier-2 mismatches.

  LogitModel     regularised logistic regression on standardised features (comp one-hot)
  BoostModel     shallow histogram gradient boosting classifier (non-linear, interactions)
  MarginModel    gradient-boosted regression for the points margin and the total; margin is
                 mapped to win/draw/loss with a Normal whose spread depends on tier and scoring
  EloLogit       logistic regression on the Elo gap alone
  Ensemble       average of the four win probabilities, plus the margin model's scoreline

Hyperparameters were confirmed on the 2008-2016 walk-forward window; 2017-2026 is reported
separately as a near-clean holdout (see the README note on what was visible during tuning).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import features as _features
from .features import COMPS
from .teams import TIER1

# Active inputs. set_features() narrows them for ablation studies; the lists are mutated in
# place so every model sees the change.
FEATURES = list(_features.FEATURES)
NUMERIC = [f for f in FEATURES if f != "comp"]


def set_features(keep):
    FEATURES[:] = [f for f in _features.FEATURES if f in keep]
    NUMERIC[:] = [f for f in FEATURES if f != "comp"]


def categorical():
    return [FEATURES.index("comp")] if "comp" in FEATURES else None


def recency_tier_weights(df, tau=None, tier1_weight=1.0):
    w = np.ones(len(df))
    if tau:
        age = (df.date.max() - df.date).dt.days.to_numpy() / 365.25
        w *= np.exp(-age / tau)
    t1 = (df.home.isin(TIER1) & df.away.isin(TIER1)).to_numpy()
    return w * np.where(t1, tier1_weight, 1.0)


def soft_labels(df, weights=None):
    """Rows, labels and weights with draws split into a half win and a half loss."""
    y = df.result.to_numpy()
    base = np.ones(len(df)) if weights is None else np.asarray(weights, float)
    draw = y == 0.5
    X = pd.concat([df, df[draw]], ignore_index=True)
    labels = np.concatenate([np.where(draw, 1, y), np.zeros(draw.sum())]).astype(int)
    w = np.concatenate([np.where(draw, 0.5, 1.0) * base, np.full(draw.sum(), 0.5) * base[draw]])
    return X, labels, w


class LogitModel:
    name = "Logistic (20 features)"

    def __init__(self, C=0.05, tau=8.0, tier1_weight=3.0):
        self.C, self.tau, self.tier1_weight = C, tau, tier1_weight

    def _design(self, df):
        num = df[NUMERIC].to_numpy(float)
        onehot = np.eye(len(COMPS))[df.comp.to_numpy(int)]
        return num, onehot

    def fit(self, df):
        X, y, w = soft_labels(df, recency_tier_weights(df, self.tau, self.tier1_weight))
        num, onehot = self._design(X)
        self.scaler = StandardScaler().fit(num)
        self.m = LogisticRegression(C=self.C, max_iter=3000)
        self.m.fit(np.hstack([self.scaler.transform(num), onehot]), y, sample_weight=w)
        return self

    def predict(self, df):
        num, onehot = self._design(df)
        return self.m.predict_proba(np.hstack([self.scaler.transform(num), onehot]))[:, 1]

    def coefficients(self):
        names = NUMERIC + [f"comp={c}" for c in COMPS]
        return pd.Series(self.m.coef_[0], index=names)


class BoostModel:
    name = "Gradient boosting (20 features)"

    def __init__(self, tau=12.0, tier1_weight=2.0, **kw):
        self.tau, self.tier1_weight = tau, tier1_weight
        # shallow trees: deeper ones (12 leaves) overfit, 0.509 v 0.501 tier-1 log-loss 2008-16
        self.kw = dict(learning_rate=0.03, max_iter=400, max_leaf_nodes=4, min_samples_leaf=100,
                       l2_regularization=1.0, random_state=0) | kw

    def fit(self, df):
        X, y, w = soft_labels(df, recency_tier_weights(df, self.tau, self.tier1_weight))
        self.m = HistGradientBoostingClassifier(categorical_features=categorical(), **self.kw)
        self.m.fit(X[FEATURES], y, sample_weight=w)
        return self

    def predict(self, df):
        return self.m.predict_proba(df[FEATURES])[:, 1]


class MarginModel:
    """Predict margin and total points; P(win) = P(Normal(margin, sd) > 0.5) etc.
    The spread is linear in (expected total, |expected margin|, both sides tier-1, one side tier-1),
    fitted to absolute residuals (E|e| = sd * sqrt(2/pi)). Tier matters most: residual sd is about
    14.5 points in tier-1 tests and 21 across all tests, while expected scoring barely moves it."""
    name = "Margin regression (20 features)"

    def __init__(self):
        kw = dict(learning_rate=0.04, max_iter=300, max_leaf_nodes=12, min_samples_leaf=40,
                  l2_regularization=1.0, categorical_features=categorical(), random_state=0)
        self.margin = HistGradientBoostingRegressor(loss="absolute_error", **kw)
        self.total = HistGradientBoostingRegressor(**kw)

    def fit(self, df):
        X = df[FEATURES]
        self.margin.fit(X, df.margin.clip(-80, 80))
        self.total.fit(X, (df.home_score + df.away_score).clip(0, 120))
        # spread from out-of-bag-ish residuals: fit on the last 30% after training on the first 70%
        cut = int(len(df) * 0.7)
        early = HistGradientBoostingRegressor(**self.margin.get_params()).fit(
            X.iloc[:cut], df.margin.iloc[:cut].clip(-80, 80))
        late = df.iloc[cut:]
        resid = np.abs(late.margin - early.predict(late[FEATURES])) * np.sqrt(np.pi / 2)
        self.sd = LinearRegression().fit(self._spread_design(late, early), resid)
        return self

    def _spread_design(self, df, margin_model=None):
        mu = (margin_model or self.margin).predict(df[FEATURES])
        both = (df.home.isin(TIER1) & df.away.isin(TIER1)).to_numpy(float)
        one = (df.home.isin(TIER1) ^ df.away.isin(TIER1)).to_numpy(float)
        return np.column_stack([self.total.predict(df[FEATURES]), np.abs(mu), both, one])

    def spread(self, df):
        return np.clip(self.sd.predict(self._spread_design(df)), 8.0, 40.0)

    def predict_dist(self, df):
        mu, sd = self.margin.predict(df[FEATURES]), self.spread(df)
        p_home = 1 - norm.cdf(0.5, mu, sd)
        p_draw = norm.cdf(0.5, mu, sd) - norm.cdf(-0.5, mu, sd)
        total = self.total.predict(df[FEATURES])
        return pd.DataFrame({"margin": mu, "sd": sd, "total": total, "p_home": p_home,
                             "p_draw": p_draw, "p_away": 1 - p_home - p_draw})

    def predict(self, df):
        d = self.predict_dist(df)
        return (d.p_home + d.p_draw / 2).to_numpy()


class Ensemble:
    name = "Ensemble"

    def __init__(self):
        self.parts = [LogitModel(), BoostModel(), MarginModel(), EloLogit()]
        self.margin_model = self.parts[2]

    def fit(self, df):
        for p in self.parts:
            p.fit(df)
        return self

    def predict_parts(self, df):
        return {p.name: p.predict(df) for p in self.parts}

    def predict(self, df):
        return np.mean(list(self.predict_parts(df).values()), axis=0)

    def predict_full(self, df):
        """Win/draw/loss, margin and expected score for each row."""
        d = self.margin_model.predict_dist(df)
        p = self.predict(df)
        # keep the margin model's draw rate, scale win/loss to the ensemble's expected result
        draw = d.p_draw.to_numpy()
        home = np.clip(p - draw / 2, 0, 1 - draw)
        out = pd.DataFrame({"p_home": home, "p_draw": draw, "p_away": 1 - home - draw,
                            "margin": d.margin, "sd": d.sd, "total": d.total})
        out["home_pts"] = (out.total + out.margin) / 2
        out["away_pts"] = (out.total - out.margin) / 2
        return out


# ---------------------------------------------------------------- baselines

class HomeRate:
    name = "Home-win base rate"

    def fit(self, df):
        self.p = df.result.mean()
        return self

    def predict(self, df):
        return np.full(len(df), self.p)


class SingleFeatureLogit:
    """Logistic regression on one rating gap: how much a single ranking system knows."""

    def __init__(self, feature, name):
        self.feature, self.name = feature, name

    def fit(self, df):
        X, y, w = soft_labels(df)
        self.m = LogisticRegression(C=10.0).fit(X[[self.feature]], y, sample_weight=w)
        return self

    def predict(self, df):
        return self.m.predict_proba(df[[self.feature]])[:, 1]


class EloLogit(SingleFeatureLogit):
    def __init__(self):
        super().__init__("elo_diff", "Elo + logistic")


class RawElo:
    name = "Elo (raw expectancy)"

    def fit(self, df):
        return self

    def predict(self, df):
        return df.elo_p.to_numpy()


def all_models():
    return [HomeRate(), SingleFeatureLogit("wr_gap", "World Rugby rankings"), RawElo(),
            EloLogit(), LogitModel(), BoostModel(), MarginModel(), Ensemble()]
