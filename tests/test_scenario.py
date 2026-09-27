import numpy as np
import pytest

import bp_scenario as sc


def test_make_separable_is_reproducible():
    X1, y1 = sc.make_separable(40, seed=3, spread=1.0, gap=4.0)
    X2, y2 = sc.make_separable(40, seed=3, spread=1.0, gap=4.0)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_make_separable_rejects_insufficient_gap():
    with pytest.raises(ValueError):
        sc.make_separable(40, seed=1, spread=1.0, gap=1.5)


def test_make_separable_always_linearly_separable():
    for seed in range(30):
        X, y = sc.make_separable(40, seed=seed, spread=1.0, gap=4.0)
        assert np.all(y * X[:, 0] > 0)


def test_make_xor_has_all_four_quadrant_labels():
    X, y = sc.make_xor(40, seed=1, spread=0.6, gap=3.0)
    signs = np.sign(X[:, 0]) * np.sign(X[:, 1])
    assert np.mean(signs == y) > 0.95


def test_build_scenario_dispatches_by_mode():
    assert sc.build_scenario("trennbar", 20, 1, 1.0, 4.0).mode == "trennbar"
    assert sc.build_scenario("xor", 20, 1, 0.6, 3.0).mode == "xor"
    with pytest.raises(ValueError):
        sc.build_scenario("unbekannt", 20, 1, 1.0, 4.0)
