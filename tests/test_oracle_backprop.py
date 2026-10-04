"""Unabhängige Orakel für den Kernalgorithmus (nicht der Code gegen sich selbst):

* Rückwärtspass gegen den Complex-Step-Gradienten (exakt bis auf Maschinengenauigkeit,
  anderer Rechenweg als die Kettenregel von Hand) auf zufälligen Netzen,
* h=0-Reduktion gegen scikit-learns Perceptron (eine Epoche, ohne Mischen),
* SGD/Momentum gegen scikit-learns Optimierer, Adam gegen die Lehrbuchformel.
"""
import numpy as np
import pytest

import bp_mlp as M
import bp_scenario as sc


def _loss_complex(params, x, y, h):
    if h == 0:
        s = np.dot(params["w2"], x) + params["b2"]
    else:
        s = np.dot(params["w2"], np.tanh(params["W1"] @ x + params["b1"])) + params["b2"]
    return -y * s  # glatter Zweig von max(0, -y*s), gilt wo y*s <= 0


def test_backward_matches_complex_step_gradient_on_random_nets():
    rng = np.random.default_rng(12345)
    checked = 0
    for _ in range(200):
        h = int(rng.integers(0, 9))
        net = M.MLP(2, h, seed=int(rng.integers(0, 10**6)))
        for k in net.params:
            net.params[k] = np.asarray(rng.normal(0, 1.0, size=np.shape(net.params[k])))
        x = rng.normal(0, 2, size=2)
        y = float(rng.choice([-1.0, 1.0]))
        s, cache = net.forward(x)
        grads = net.backward(y, s, cache)
        if y * s > 0:
            assert grads is None
            continue
        checked += 1
        for key, g in grads.items():
            ref = np.zeros(np.shape(net.params[key]))
            for idx in np.ndindex(*ref.shape) if ref.shape else [()]:
                p = {k: np.array(v, dtype=complex) for k, v in net.params.items()}
                p[key][idx] += 1e-30j
                ref[idx] = _loss_complex(p, x, y, h).imag / 1e-30
            np.testing.assert_allclose(np.asarray(g), ref, atol=1e-9)
    assert checked > 50


def test_h0_training_equals_sklearn_perceptron_epoch():
    linear_model = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(3)
    for _ in range(40):
        n = int(rng.integers(4, 40))
        seed = int(rng.integers(0, 10**5))
        gap = float(rng.uniform(2.1, 8))
        eta = float(rng.uniform(0.05, 1.0))
        X, y = sc.make_separable(n, seed, 1.0, gap)
        clf = linear_model.Perceptron(penalty=None, alpha=0.0, max_iter=1, tol=None,
                                      shuffle=False, eta0=eta).fit(X, y)
        net = M.MLP(2, 0, seed=0)
        net.params["w2"] = np.zeros(2)
        net.params["b2"] = np.zeros(())
        M.train_epoch(net, M.SGD(eta), X, y)
        np.testing.assert_allclose(net.params["w2"], clf.coef_[0], atol=1e-9)
        assert abs(float(net.params["b2"]) - clf.intercept_[0]) < 1e-9


def test_optimizers_match_independent_references():
    sk = pytest.importorskip("sklearn.neural_network._stochastic_optimizers")
    rng = np.random.default_rng(7)
    for _ in range(30):
        shapes = [(int(rng.integers(1, 4)), int(rng.integers(1, 4))), (3,)]
        p0 = [rng.normal(size=s) for s in shapes]
        eta = float(rng.uniform(0.01, 1.0))
        seq = [[rng.normal(size=s) * 3 for s in shapes] for _ in range(int(rng.integers(1, 25)))]
        for name, mom in (("sgd", 0.0), ("momentum", 0.9)):
            params = {f"a{i}": p0[i].copy() for i in range(2)}
            opt = M.make_optimizer(name, eta)
            ref = sk.SGDOptimizer([q.copy() for q in p0], learning_rate_init=eta,
                                  lr_schedule="constant", momentum=mom, nesterov=False)
            refp = [q.copy() for q in p0]
            for gs in seq:
                opt.step(params, {f"a{i}": gs[i] for i in range(2)})
                ref.update_params(refp, [g.copy() for g in gs])
            for i in range(2):
                np.testing.assert_allclose(params[f"a{i}"], refp[i], atol=1e-9)
        # Adam: Lehrbuchformel (Kingma & Ba 2015, Alg. 1) mit Bias-Korrektur
        params = {f"a{i}": p0[i].copy() for i in range(2)}
        opt = M.make_optimizer("adam", eta)
        pk = [q.copy() for q in p0]
        mk = [np.zeros_like(q) for q in p0]
        vk = [np.zeros_like(q) for q in p0]
        for t, gs in enumerate(seq, 1):
            opt.step(params, {f"a{i}": gs[i] for i in range(2)})
            for i in range(2):
                mk[i] = 0.9 * mk[i] + 0.1 * gs[i]
                vk[i] = 0.999 * vk[i] + 0.001 * gs[i] ** 2
                pk[i] = pk[i] - eta * (mk[i] / (1 - 0.9 ** t)) / (
                    np.sqrt(vk[i] / (1 - 0.999 ** t)) + 1e-8)
        for i in range(2):
            np.testing.assert_allclose(params[f"a{i}"], pk[i], atol=1e-12)
