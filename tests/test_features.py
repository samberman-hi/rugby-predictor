import numpy as np
import pandas as pd

from rugby import features


def toy_matches(n=60, seed=0):
    rng = np.random.default_rng(seed)
    sides = ["IRE", "NZL", "FRA", "RSA", "JPN", "FIJ"]
    rows = []
    for i in range(n):
        h, a = rng.choice(sides, 2, replace=False)
        rows.append(dict(date=pd.Timestamp("2015-01-03") + pd.Timedelta(days=7 * i), home=h, away=a,
                         home_score=float(rng.integers(0, 40)), away_score=float(rng.integers(0, 40)),
                         home_tries=float(rng.integers(0, 5)), away_tries=float(rng.integers(0, 5)),
                         comp="test", knockout=False, host=h, venue_country=1))
    return pd.DataFrame(rows)


def test_feature_set_has_twenty_inputs():
    assert len(features.FEATURES) == 20
    f = features.build(toy_matches())
    assert f[features.FEATURES].notna().all().all()


def test_no_look_ahead():
    """Changing results after match k cannot change any feature up to and including match k."""
    m = toy_matches()
    k = 40
    changed = m.copy()
    changed.loc[k:, "home_score"] = 99.0
    changed.loc[k:, "away_score"] = 0.0
    a = features.build(m)[features.FEATURES]
    b = features.build(changed)[features.FEATURES]
    pd.testing.assert_frame_equal(a.iloc[:k + 1], b.iloc[:k + 1])
    assert not a.iloc[k + 1:].equals(b.iloc[k + 1:])


def test_fixtures_get_features_but_never_update_ratings():
    m = toy_matches()
    fx = pd.DataFrame([dict(date=pd.Timestamp("2016-06-01"), home="IRE", away="NZL", home_score=np.nan,
                            away_score=np.nan, home_tries=np.nan, away_tries=np.nan, comp="test",
                            knockout=False, host="IRE", venue_country=1)])
    with_fx = features.build(m, fx)
    without = features.build(m)
    assert with_fx.attrs["state"]["elo"].r == without.attrs["state"]["elo"].r
    row = with_fx[~with_fx.played].iloc[0]
    assert row.venue == 1 and row.elo_diff != 0


def test_context_features():
    assert features.offseason("IRE", "NZL", 7) == 1   # northern side touring south in July
    assert features.offseason("NZL", "IRE", 11) == 1  # southern side touring north in November
    assert features.offseason("NZL", "IRE", 7) == 0
    assert features.km("IRE", "NZL") > 18000
    assert features.tz_hours("IRE", "NZL") > 11
