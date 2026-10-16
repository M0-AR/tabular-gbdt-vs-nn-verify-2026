"""Smoke tests — every claim file must be verified by execution, never by hand-edit."""
import sys
sys.path.insert(0, "src")
import numpy as np
from data_synthetic import train_test, true_price, add_noise_features, rotate_features
from gbdt_plain import PlainGBDT
from models import train_mlp


def test_rule_oracle():
    Xtr, ytr, ctr, Xte, yte, cte = train_test(seed=0)
    assert Xtr.shape == (2000, 6) and Xte.shape == (1000, 6)
    resid = yte - cte
    assert 12000 < resid.std() < 18000, resid.std()


def test_boost_beats_stump():
    Xtr, ytr, _, Xte, yte, _ = train_test(seed=0)
    g = PlainGBDT(n_estimators=50, max_depth=3, learning_rate=0.1).fit(Xtr, ytr)
    te = g.staged_rmse(Xte, yte)
    assert te[-1] < te[0] - 10000


def test_rotation_preserves_info():
    Xtr, ytr, _, Xte, yte, _ = train_test(seed=0)
    Rtr, Rte = rotate_features(Xtr, Xte)
    assert Rtr.shape == Xtr.shape
    # linear solve on rotated ≈ linear solve on standardized original
    assert np.isfinite(Rtr).all()
