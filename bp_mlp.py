"""Ein-Schicht-MLP (tanh-Verdeckte + linearer Ausgang) mit Backpropagation von
Hand, Perceptron-Criterion-Verlust und drei Optimierern (SGD/Momentum/Adam).

Bei n_hidden=0 ist dies algebraisch IDENTISCH zur klassischen Perceptron-Regel
aus Stueck 1 (Rosenblatt 1958): der Sub-Gradient von L=max(0,-y*s) bei
Fehlklassifikation ist genau -y*x, ein SGD-Schritt damit w <- w + eta*y*x -
keine Naeherung, sondern dieselbe Formel (siehe tests/test_mlp.py)."""
from dataclasses import dataclass, field

import numpy as np


class MLP:
    def __init__(self, n_in: int, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_hidden = n_hidden
        if n_hidden > 0:
            limit1 = np.sqrt(6.0 / (n_in + n_hidden))  # Xavier (tanh)
            self.params = {
                "W1": rng.uniform(-limit1, limit1, size=(n_hidden, n_in)),
                "b1": np.zeros(n_hidden),
                "w2": rng.uniform(-np.sqrt(6.0 / (n_hidden + 1)),
                                  np.sqrt(6.0 / (n_hidden + 1)), size=n_hidden),
                "b2": np.zeros(()),
            }
        else:
            limit = np.sqrt(6.0 / (n_in + 1))
            self.params = {
                "w2": rng.uniform(-limit, limit, size=n_in),
                "b2": np.zeros(()),
            }

    def forward(self, x: np.ndarray):
        p = self.params
        if self.n_hidden > 0:
            z1 = p["W1"] @ x + p["b1"]
            h = np.tanh(z1)
            s = p["w2"] @ h + p["b2"]
            return float(s), (x, h)
        s = p["w2"] @ x + p["b2"]
        return float(s), (x, None)

    def backward(self, y: float, s: float, cache):
        """Sub-Gradient von L = max(0, -y*s). Bei y*s > 0 (korrekt klassifiziert,
        Marge > 0): kein Fehler, kein Gradient (None). Bei y*s <= 0: dL/ds = -y."""
        if y * s > 0:
            return None
        x, h = cache
        ds = -y
        if self.n_hidden > 0:
            dw2 = ds * h
            db2 = ds
            dh = ds * self.params["w2"]
            dz1 = dh * (1 - h ** 2)
            dW1 = np.outer(dz1, x)
            db1 = dz1
            return {"W1": dW1, "b1": db1, "w2": dw2, "b2": db2}
        return {"w2": ds * x, "b2": ds}

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.array([self.forward(x)[0] for x in X])
        return np.where(scores >= 0, 1.0, -1.0)


# ---------- Optimierer ----------

class SGD:
    def __init__(self, eta: float = 0.5):
        self.eta = eta

    def step(self, params: dict, grads: dict) -> None:
        for key, g in grads.items():
            params[key] = params[key] - self.eta * g


class Momentum:
    def __init__(self, eta: float = 0.5, mu: float = 0.9):
        self.eta = eta
        self.mu = mu
        self.velocity: dict = {}

    def step(self, params: dict, grads: dict) -> None:
        for key, g in grads.items():
            v = self.velocity.get(key, np.zeros_like(g))
            v = self.mu * v - self.eta * g
            self.velocity[key] = v
            params[key] = params[key] + v


class Adam:
    def __init__(self, eta: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict = {}
        self.v: dict = {}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for key, g in grads.items():
            m = self.beta1 * self.m.get(key, np.zeros_like(g)) + (1 - self.beta1) * g
            v = self.beta2 * self.v.get(key, np.zeros_like(g)) + (1 - self.beta2) * g ** 2
            self.m[key] = m
            self.v[key] = v
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            params[key] = params[key] - self.eta * m_hat / (np.sqrt(v_hat) + self.eps)


def make_optimizer(name: str, eta: float):
    if name == "sgd":
        return SGD(eta)
    if name == "momentum":
        return Momentum(eta)
    if name == "adam":
        return Adam(eta)
    raise ValueError(f"unbekannter Optimierer: {name!r}")


@dataclass
class FitResult:
    converged: bool
    epochs: int
    errors_per_epoch: list = field(default_factory=list)


def train_epoch(mlp: MLP, optimizer, X: np.ndarray, y: np.ndarray) -> int:
    n_errors = 0
    for xi, yi in zip(X, y):
        s, cache = mlp.forward(xi)
        grads = mlp.backward(yi, s, cache)
        if grads is not None:
            n_errors += 1
            optimizer.step(mlp.params, grads)
    return n_errors


def fit(mlp: MLP, optimizer, X: np.ndarray, y: np.ndarray, max_epochs: int = 500) -> FitResult:
    errors_per_epoch = []
    for epoch in range(max_epochs):
        n_errors = train_epoch(mlp, optimizer, X, y)
        errors_per_epoch.append(n_errors)
        if n_errors == 0:
            return FitResult(True, epoch + 1, errors_per_epoch)
    return FitResult(False, max_epochs, errors_per_epoch)
