import bp_evaluation as ev


def test_hidden_unit_sweep_matches_lehrbuchantwort():
    rows = ev.hidden_unit_sweep(sizes=(0, 1, 2, 3), n_inits=15)
    rate_by_h = {r["hidden"]: r["rate"] for r in rows}
    assert rate_by_h[0] == 0.0
    assert rate_by_h[1] == 0.0
    assert 0.0 < rate_by_h[2] < 1.0  # gelingt manchmal, nicht garantiert
    assert rate_by_h[3] > rate_by_h[2]  # mehr Einheiten helfen


def test_reduction_check_is_exact():
    out = ev.reduction_check()
    assert out["identical"]


def test_optimizer_comparison_all_optimizers_present():
    rows = ev.optimizer_comparison(n_seeds=5, max_epochs=1000)
    names = {r["optimizer"] for r in rows}
    assert names == {"sgd", "momentum", "adam"}
    for row in rows:
        assert row["n_converged"] >= 0
