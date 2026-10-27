import numpy as np
import pytest

from rugby import evaluation as ev


def test_metrics():
    y = np.array([1, 0, 0.5, 1])
    p = np.array([0.8, 0.3, 0.5, 0.4])
    assert ev.brier(p, y) == pytest.approx(np.mean((p - y) ** 2))
    assert ev.accuracy(p, y) == pytest.approx(2 / 3)  # the draw is excluded
    assert ev.log_loss([0.5] * 4, y) == pytest.approx(np.log(2))


def test_calibration_bins_cover_all_rows():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < p).astype(float)
    cal = ev.calibration(p, y)
    assert cal.n.sum() == 500
    assert np.all(np.abs(cal.pred - cal.obs) < 0.2)
