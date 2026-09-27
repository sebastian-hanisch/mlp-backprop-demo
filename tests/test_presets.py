import bp_constants as C
import bp_evaluation as ev
import bp_presets as pr


def test_all_presets_have_valid_settings():
    for key, preset in C.PRESETS.items():
        assert preset["mode"] in C.MODES
        b = pr.bounds(preset["mode"])
        assert b["gap_min"] <= preset["gap"] <= b["gap_max"]
        assert C.HIDDEN_MIN <= preset["hidden"] <= C.HIDDEN_MAX
        assert preset["optimizer"] in C.OPTIMIZERS


def test_preset_perceptron_reduces_to_perceptron_rule():
    p = C.PRESETS["perceptron"]
    assert p["hidden"] == 0
    out = ev.reduction_check()
    assert out["identical"]


def test_preset_xor_gelingt_can_converge():
    p = C.PRESETS["xor_gelingt"]
    settings = ev.Settings(p["mode"], p["n"], p["seed"], p["eta"], p["hidden"],
                           p["max_epochs"], p["optimizer"], p["gap"])
    out = ev.analyse(settings)
    assert out["result"].converged


def test_preset_xor_scheitert_never_converges():
    p = C.PRESETS["xor_scheitert"]
    settings = ev.Settings(p["mode"], p["n"], p["seed"], p["eta"], p["hidden"],
                           p["max_epochs"], p["optimizer"], p["gap"])
    out = ev.analyse(settings)
    assert not out["result"].converged
