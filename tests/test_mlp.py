import numpy as np

import bp_evaluation as ev
import bp_mlp as mlp_mod
import bp_scenario as sc


def test_forward_shapes_with_hidden_units():
    net = mlp_mod.MLP(2, 3, seed=0)
    s, cache = net.forward(np.array([1.0, -1.0]))
    assert isinstance(s, float)
    x, h = cache
    assert h.shape == (3,)


def test_forward_without_hidden_units_is_linear():
    net = mlp_mod.MLP(2, 0, seed=0)
    net.params["w2"] = np.array([2.0, -1.0])
    net.params["b2"] = np.asarray(0.5)
    s, _ = net.forward(np.array([1.0, 1.0]))
    assert np.isclose(s, 2.0 * 1.0 - 1.0 * 1.0 + 0.5)


def test_backward_returns_none_when_correctly_classified_with_margin():
    net = mlp_mod.MLP(2, 0, seed=0)
    net.params["w2"] = np.array([1.0, 0.0])
    net.params["b2"] = np.asarray(0.0)
    x = np.array([1.0, 0.0])
    s, cache = net.forward(x)  # s=1.0
    grads = net.backward(1.0, s, cache)  # y*s = 1.0 > 0
    assert grads is None


def test_backward_returns_gradient_on_misclassification():
    net = mlp_mod.MLP(2, 0, seed=0)
    net.params["w2"] = np.array([0.0, 0.0])
    net.params["b2"] = np.asarray(0.0)
    x = np.array([1.0, 2.0])
    s, cache = net.forward(x)  # s=0
    grads = net.backward(1.0, s, cache)  # y*s = 0 <= 0 -> Fehler
    assert grads is not None
    np.testing.assert_array_almost_equal(grads["w2"], -1.0 * x)  # ds=-y=-1.0 -> dw2=ds*x


def test_gradient_check_passes_below_1e_minus_8():
    err = ev.gradient_check(hidden=3, seed=5)
    assert err < 1e-8


def test_gradient_check_passes_with_zero_hidden_units():
    err = ev.gradient_check(hidden=0, seed=5)
    assert err < 1e-8


def test_hidden_units_0_and_1_never_solve_xor():
    X, y = sc.make_xor(40, seed=1, spread=0.6, gap=3.0)
    for h in (0, 1):
        for seed in range(10):
            net = mlp_mod.MLP(2, h, seed=seed)
            optimizer = mlp_mod.SGD(0.5)
            result = mlp_mod.fit(net, optimizer, X, y, max_epochs=500)
            assert not result.converged, f"h={h} seed={seed} sollte nicht konvergieren"


def test_hidden_units_4_almost_always_solves_xor():
    X, y = sc.make_xor(40, seed=1, spread=0.6, gap=3.0)
    successes = 0
    for seed in range(20):
        net = mlp_mod.MLP(2, 4, seed=seed)
        optimizer = mlp_mod.SGD(0.5)
        result = mlp_mod.fit(net, optimizer, X, y, max_epochs=500)
        if result.converged:
            successes += 1
    assert successes >= 18  # >=90%


def test_optimizers_reduce_mean_error_count_over_time():
    """Nicht-konvexes Problem: ein EINZELNER Lauf kann zwischendurch schwanken
    (siehe Momentum/Adam-Überschwinger), daher eine statistische statt eine
    monoton-je-Lauf-Behauptung ueber mehrere Seeds."""
    X, y = sc.make_xor(40, seed=1, spread=0.6, gap=3.0)
    for opt_name in ("sgd", "momentum", "adam"):
        first_errs, last_errs = [], []
        for seed in range(10):
            net = mlp_mod.MLP(2, 2, seed=seed)
            optimizer = mlp_mod.make_optimizer(opt_name, 0.1)
            first_errs.append(mlp_mod.train_epoch(net, optimizer, X, y))
            for _ in range(30):
                errs = mlp_mod.train_epoch(net, optimizer, X, y)
            last_errs.append(errs)
        assert sum(last_errs) / len(last_errs) < sum(first_errs) / len(first_errs)
