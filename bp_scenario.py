"""Datengeneratoren fuer die zwei Vehikel-Modi: linear trennbar und XOR-Muster.
Eigenstaendige Kopie der Konstruktion aus perceptron-demo (kein Cross-Repo-Import
- jedes Demo-Repo bleibt fuer sich deploybar). Beschraenktes Scheiben-Rauschen
statt Gauss-Schwaenze: garantiert Trennbarkeit im Modus 'trennbar' konstruktiv
(gap > 2*spread), mit exakt bekannter Marge."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    mode: str
    X: np.ndarray
    y: np.ndarray
    seed: int
    n: int
    spread: float
    gap: float


def _disc_noise(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def make_separable(n: int, seed: int, spread: float, gap: float):
    if gap <= 2 * spread:
        raise ValueError(
            f"gap ({gap}) muss > 2*spread ({2*spread}) sein, sonst ist "
            "Trennbarkeit nicht garantiert."
        )
    rng = np.random.default_rng(seed)
    n_pos, n_neg = n // 2, n - n // 2
    pos = np.array([gap / 2, 0.0]) + _disc_noise(rng, n_pos, spread)
    neg = np.array([-gap / 2, 0.0]) + _disc_noise(rng, n_neg, spread)
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n_pos), -np.ones(n_neg)])
    order = rng.permutation(len(y))
    return X[order], y[order]


def make_xor(n: int, seed: int, spread: float, gap: float):
    rng = np.random.default_rng(seed)
    n_each = n // 4
    centers = [(gap, gap), (-gap, -gap), (gap, -gap), (-gap, gap)]
    labels = [1, 1, -1, -1]
    xs, ys = [], []
    for c, lab in zip(centers, labels):
        xs.append(np.array(c) + _disc_noise(rng, n_each, spread))
        ys.append(np.full(n_each, lab))
    X = np.vstack(xs)
    y = np.concatenate(ys)
    order = rng.permutation(len(y))
    return X[order], y[order]


def build_scenario(mode: str, n: int, seed: int, spread: float, gap: float) -> Scenario:
    if mode == "trennbar":
        X, y = make_separable(n, seed, spread, gap)
    elif mode == "xor":
        X, y = make_xor(n, seed, spread, gap)
    else:
        raise ValueError(f"unbekannter Modus: {mode!r}")
    return Scenario(mode=mode, X=X, y=y, seed=seed, n=n, spread=spread, gap=gap)


def exact_margin_trennbar(spread: float, gap: float) -> float:
    return gap / 2 - spread
