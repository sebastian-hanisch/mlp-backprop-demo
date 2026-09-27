"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

DEFAULT_SEED = 0

MODE_TRENNBAR = "trennbar"
MODE_XOR = "xor"
MODES = (MODE_TRENNBAR, MODE_XOR)
MODE_LABELS = {MODE_TRENNBAR: "Linear trennbar", MODE_XOR: "XOR-Muster"}

N_MIN, N_MAX, N_DEFAULT = 10, 200, 40
ETA_MIN, ETA_MAX, ETA_DEFAULT = 0.05, 1.0, 0.5
MAX_EPOCHS_MIN, MAX_EPOCHS_MAX, MAX_EPOCHS_DEFAULT = 50, 2000, 500

HIDDEN_MIN, HIDDEN_MAX, HIDDEN_DEFAULT = 0, 8, 2

SPREAD_TRENNBAR = 1.0
SPREAD_XOR = 0.6
GAP_TRENNBAR_MIN, GAP_TRENNBAR_MAX, GAP_TRENNBAR_DEFAULT = 2.02, 8.0, 4.0
GAP_XOR_MIN, GAP_XOR_MAX, GAP_XOR_DEFAULT = 1.5, 5.0, 3.0

OPTIMIZERS = ("sgd", "momentum", "adam")
OPTIMIZER_LABELS = {"sgd": "SGD", "momentum": "Momentum", "adam": "Adam"}

# Hidden-Unit-Sweep (fuer den 📐-Abschnitt), billig genug fuer eager statt
# on-demand Berechnung
HIDDEN_SWEEP_SIZES = (0, 1, 2, 3, 4, 6, 8)
HIDDEN_SWEEP_INITS = 30
HIDDEN_SWEEP_MAX_EPOCHS = 500

PRESETS = {
    "perceptron": dict(
        label="0 verdeckte Einheiten = Perceptron",
        mode=MODE_TRENNBAR, n=40, gap=4.0, eta=1.0, hidden=0, max_epochs=500,
        optimizer="sgd", seed=7,
        help="Ohne verdeckte Schicht ist dies exakt die Perceptron-Regel aus Stück 1 "
             "(gleiche Update-Formel, gleiche Trajektorie).",
    ),
    "xor_gelingt": dict(
        label="2 verdeckte Einheiten löst XOR",
        mode=MODE_XOR, n=40, gap=3.0, eta=0.5, hidden=2, max_epochs=500,
        optimizer="sgd", seed=16,
        help="Die kleinste Größe, die XOR überhaupt lösen KANN (nicht immer - nur "
             "bei rund 30 % der Zufalls-Initialisierungen).",
    ),
    "xor_scheitert": dict(
        label="1 verdeckte Einheit scheitert",
        mode=MODE_XOR, n=40, gap=3.0, eta=0.5, hidden=1, max_epochs=500,
        optimizer="sgd", seed=3,
        help="Wie das Perceptron: strukturell zu wenig, um XOR zu trennen - keine "
             "Initialisierung erreicht 0 Trainingsfehler.",
    ),
}
