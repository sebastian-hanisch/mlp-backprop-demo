"""Kennzahlen und Sweeps: Settings-Dataclass, analyse()-Einstiegspunkt,
Hidden-Unit-Sweep, Gradienten-Check, Reduktions-Check auf die Perceptron-Regel,
Optimierer-Vergleich."""
from dataclasses import dataclass

import numpy as np

import bp_constants as C
import bp_mlp as mlp_mod
import bp_scenario as sc


@dataclass(frozen=True)
class Settings:
    mode: str
    n: int
    seed: int
    eta: float
    hidden: int
    max_epochs: int
    optimizer: str
    gap: float


def _spread_for(mode: str) -> float:
    return C.SPREAD_TRENNBAR if mode == C.MODE_TRENNBAR else C.SPREAD_XOR


def analyse(settings: Settings) -> dict:
    spread = _spread_for(settings.mode)
    scenario = sc.build_scenario(settings.mode, settings.n, settings.seed, spread, settings.gap)
    net = mlp_mod.MLP(2, settings.hidden, seed=settings.seed)
    optimizer = mlp_mod.make_optimizer(settings.optimizer, settings.eta)
    result = mlp_mod.fit(net, optimizer, scenario.X, scenario.y, max_epochs=settings.max_epochs)
    return {"scenario": scenario, "result": result, "net": net}


def hidden_unit_sweep(
    sizes=C.HIDDEN_SWEEP_SIZES,
    n_inits=C.HIDDEN_SWEEP_INITS,
    n=C.N_DEFAULT,
    seed_data=1,
    mode=C.MODE_XOR,
    gap=C.GAP_XOR_DEFAULT,
    eta=C.ETA_DEFAULT,
    max_epochs=C.HIDDEN_SWEEP_MAX_EPOCHS,
) -> list:
    """Fuer jede Groesse h: Anteil zufaelliger Initialisierungen, die auf 0
    Trainingsfehler kommen (dasselbe Datenset ueber alle Inits, nur die
    Gewichts-Initialisierung variiert)."""
    spread = _spread_for(mode)
    scenario = sc.build_scenario(mode, n, seed_data, spread, gap)
    rows = []
    for h in sizes:
        successes = 0
        for init_seed in range(n_inits):
            net = mlp_mod.MLP(2, h, seed=init_seed)
            optimizer = mlp_mod.SGD(eta)
            result = mlp_mod.fit(net, optimizer, scenario.X, scenario.y, max_epochs=max_epochs)
            if result.converged:
                successes += 1
        rows.append({"hidden": h, "successes": successes, "n_inits": n_inits,
                      "rate": successes / n_inits})
    return rows


def gradient_check(hidden: int = 3, seed: int = 5, eps: float = 1e-6) -> float:
    """Backprop-Gradient gegen finite Differenzen, an einem Punkt mit
    erzwungenem Fehler (Marge < 0, sonst gibt es keinen Gradienten zu pruefen).
    Gibt den maximalen relativen Fehler ueber alle Parameter zurueck. Jeder
    Parameter wird als flaches Array behandelt (auch der 0-d Bias b2), damit
    dieselbe Schleife fuer Skalare wie fuer Matrizen funktioniert."""
    net = mlp_mod.MLP(2, hidden, seed=seed)
    x = np.array([0.7, -0.3])
    s0, cache = net.forward(x)
    y = -1.0 if s0 >= 0 else 1.0  # erzwingt y*s <= 0
    grads = net.backward(y, s0, cache)

    def loss() -> float:
        s, _ = net.forward(x)
        return max(0.0, -y * s)

    max_rel_err = 0.0
    for key, grad in grads.items():
        param = net.params[key]
        grad_flat = np.atleast_1d(grad).ravel()
        flat_view = param.reshape(-1) if param.shape else None
        n_entries = grad_flat.size
        for flat_idx in range(n_entries):
            if flat_view is not None:
                orig = flat_view[flat_idx]
                flat_view[flat_idx] = orig + eps
                l_plus = loss()
                flat_view[flat_idx] = orig - eps
                l_minus = loss()
                flat_view[flat_idx] = orig
            else:  # 0-d Skalar (b2 bei hidden=0)
                orig = float(param)
                net.params[key] = np.asarray(orig + eps)
                l_plus = loss()
                net.params[key] = np.asarray(orig - eps)
                l_minus = loss()
                net.params[key] = np.asarray(orig)
            numeric = (l_plus - l_minus) / (2 * eps)
            analytic = float(grad_flat[flat_idx])
            denom = max(abs(numeric), abs(analytic), 1e-12)
            max_rel_err = max(max_rel_err, abs(numeric - analytic) / denom)
    return max_rel_err


def perceptron_reference(X: np.ndarray, y: np.ndarray, eta: float):
    """Die klassische Perceptron-Regel aus Stueck 1, unabhaengig neu
    implementiert (kein Cross-Repo-Import) - Referenz fuer den Reduktions-Test."""
    w = np.zeros(X.shape[1])
    b = 0.0
    for xi, yi in zip(X, y):
        margin = yi * (w @ xi + b)
        if margin <= 0:
            w = w + eta * yi * xi
            b = b + eta * yi
    return w, b


def reduction_check(n=30, seed=7, spread=1.0, gap=4.0, eta=1.0) -> dict:
    """MLP mit hidden=0 + Perceptron-Criterion-Verlust + SGD muss exakt
    dieselbe Trajektorie wie die klassische Perceptron-Regel durchlaufen."""
    X, y = sc.make_separable(n, seed, spread, gap)
    w_ref, b_ref = perceptron_reference(X, y, eta)

    net = mlp_mod.MLP(2, 0, seed=0)
    net.params["w2"] = np.zeros(2)
    net.params["b2"] = np.zeros(())
    optimizer = mlp_mod.SGD(eta)
    for xi, yi in zip(X, y):
        s, cache = net.forward(xi)
        grads = net.backward(yi, s, cache)
        if grads is not None:
            optimizer.step(net.params, grads)

    identical = np.allclose(w_ref, net.params["w2"]) and np.isclose(b_ref, float(net.params["b2"]))
    return {"w_perceptron": w_ref, "b_perceptron": b_ref,
            "w_mlp": net.params["w2"], "b_mlp": float(net.params["b2"]),
            "identical": identical}


def optimizer_comparison(
    optimizers=C.OPTIMIZERS, n_seeds=10, hidden=2, mode=C.MODE_XOR,
    gap=C.GAP_XOR_DEFAULT, eta=0.1, max_epochs=2000, n=C.N_DEFAULT, seed_data=1,
) -> list:
    spread = _spread_for(mode)
    scenario = sc.build_scenario(mode, n, seed_data, spread, gap)
    rows = []
    for opt_name in optimizers:
        epochs_list = []
        n_converged = 0
        for seed in range(n_seeds):
            net = mlp_mod.MLP(2, hidden, seed=seed)
            optimizer = mlp_mod.make_optimizer(opt_name, eta)
            result = mlp_mod.fit(net, optimizer, scenario.X, scenario.y, max_epochs=max_epochs)
            if result.converged:
                n_converged += 1
                epochs_list.append(result.epochs)
        rows.append({
            "optimizer": opt_name, "n_converged": n_converged, "n_seeds": n_seeds,
            "mean_epochs": float(np.mean(epochs_list)) if epochs_list else None,
        })
    return rows
