import numpy as np

import bp_constants as C
import bp_handcrafted as hc
import bp_scenario as sc


def test_handcrafted_net_achieves_perfect_accuracy_on_xor():
    net = hc.build_handcrafted_xor_net()
    X, y = sc.make_xor(n=40, seed=1, spread=C.SPREAD_XOR, gap=C.GAP_XOR_DEFAULT)
    X_norm = X / C.GAP_XOR_DEFAULT
    preds = net.predict(X_norm)
    assert np.mean(preds == y) == 1.0


def test_handcrafted_net_generalises_across_seeds():
    net = hc.build_handcrafted_xor_net()
    for seed in range(10):
        X, y = sc.make_xor(n=40, seed=seed, spread=C.SPREAD_XOR, gap=C.GAP_XOR_DEFAULT)
        X_norm = X / C.GAP_XOR_DEFAULT
        preds = net.predict(X_norm)
        assert np.mean(preds == y) == 1.0
