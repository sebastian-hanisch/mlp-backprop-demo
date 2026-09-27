"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen
neu berechnet, mit denselben Presets/Seeds wie im README zitiert."""
import bp_constants as C
import bp_evaluation as ev


def test_claim_hidden_unit_sweep_matches_readme_table():
    rows = ev.hidden_unit_sweep()
    expected = {0: 0, 1: 0, 2: 9, 3: 30, 4: 29, 6: 30, 8: 30}
    for row in rows:
        assert row["successes"] == expected[row["hidden"]]
        assert row["n_inits"] == 30


def test_claim_gradient_check_below_1e_minus_9():
    assert ev.gradient_check(hidden=3, seed=5) < 1e-9


def test_claim_reduction_check_is_exact_match():
    out = ev.reduction_check()
    assert out["identical"]
    assert round(out["w_perceptron"][0], 5) == round(out["w_mlp"][0], 5)
    assert round(out["b_perceptron"], 5) == round(out["b_mlp"], 5)


def test_claim_handcrafted_xor_solution_is_perfect():
    import bp_handcrafted as hc
    import bp_scenario as sc
    import numpy as np
    net = hc.build_handcrafted_xor_net()
    X, y = sc.make_xor(n=40, seed=1, spread=C.SPREAD_XOR, gap=C.GAP_XOR_DEFAULT)
    preds = net.predict(X / C.GAP_XOR_DEFAULT)
    assert np.mean(preds == y) == 1.0


def test_claim_optimizer_comparison_momentum_underperforms_sgd():
    """Echter, nicht-offensichtlicher Befund: bei gleicher Lernrate konvergiert
    Momentum SCHLECHTER als einfaches SGD (Oszillation durch konstant-grosse
    Subgradienten + hohe Massentraegheit) - keine Behauptung, gemessen."""
    rows = ev.optimizer_comparison()
    by_name = {r["optimizer"]: r for r in rows}
    assert by_name["sgd"]["n_converged"] == 7
    assert by_name["momentum"]["n_converged"] == 0
    assert by_name["adam"]["n_converged"] == 9


def test_claim_preset_perceptron_converges_in_2_epochs():
    p = C.PRESETS["perceptron"]
    settings = ev.Settings(p["mode"], p["n"], p["seed"], p["eta"], p["hidden"],
                           p["max_epochs"], p["optimizer"], p["gap"])
    out = ev.analyse(settings)
    assert out["result"].converged
    assert out["result"].epochs == 2


def test_claim_preset_xor_gelingt_converges_in_15_epochs():
    p = C.PRESETS["xor_gelingt"]
    settings = ev.Settings(p["mode"], p["n"], p["seed"], p["eta"], p["hidden"],
                           p["max_epochs"], p["optimizer"], p["gap"])
    out = ev.analyse(settings)
    assert out["result"].converged
    assert out["result"].epochs == 15


def test_claim_preset_xor_scheitert_never_converges_within_500_epochs():
    p = C.PRESETS["xor_scheitert"]
    settings = ev.Settings(p["mode"], p["n"], p["seed"], p["eta"], p["hidden"],
                           p["max_epochs"], p["optimizer"], p["gap"])
    out = ev.analyse(settings)
    assert not out["result"].converged
    assert out["result"].epochs == 500
