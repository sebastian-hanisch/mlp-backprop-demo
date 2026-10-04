from streamlit.testing.v1 import AppTest


def _fresh():
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    return at


def test_app_runs_without_exception():
    at = _fresh()
    assert not at.exception


def test_footer_is_present():
    at = _fresh()
    captions = [c.value for c in at.caption]
    assert any("Sebastian Hanisch" in c and "Über mich" in c for c in captions)


def test_default_mode_is_xor():
    at = _fresh()
    radios = [r for r in at.radio if r.label == "Datenmodus"]
    assert radios[0].value == "xor"


def test_preset_buttons_exist_and_perceptron_preset_works():
    at = _fresh()
    labels = [b.label for b in at.button]
    assert "0 verdeckte Einheiten = Perceptron" in labels
    perceptron_button = [b for b in at.button if b.label == "0 verdeckte Einheiten = Perceptron"][0]
    perceptron_button.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Konvergiert?"] == "Ja"
    assert metrics["Verdeckte Einheiten"] == "0"


def test_xor_gelingt_preset_converges():
    at = _fresh()
    btn = [b for b in at.button if b.label == "2 verdeckte Einheiten löst XOR"][0]
    btn.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Konvergiert?"] == "Ja"


def test_xor_scheitert_preset_does_not_converge():
    at = _fresh()
    btn = [b for b in at.button if b.label == "1 verdeckte Einheit scheitert"][0]
    btn.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert "Nein" in metrics["Konvergiert?"]


def test_hidden_slider_extreme_values_do_not_crash():
    at = _fresh()
    hidden_slider = [s for s in at.slider if s.label.startswith("Verdeckte")][0]
    hidden_slider.set_value(hidden_slider.min).run()
    assert not at.exception
    hidden_slider = [s for s in at.slider if s.label.startswith("Verdeckte")][0]
    hidden_slider.set_value(hidden_slider.max).run()
    assert not at.exception


def test_optimizer_selectbox_can_switch_without_crash():
    # je ein frischer App-Lauf pro Optimierer statt mehrerer set_value().run()
    # auf demselben "at": der Epochen-Schieberegler traegt den Optimierer im
    # Widget-Key (damit er bei Konfigurationswechsel neu einrastet statt einen
    # jetzt ungueltigen alten Wert zu behalten) - das laesst seinen ALTEN
    # Widget-Knoten im internen Baum zurueck, was die AppTest-Testinfrastruktur
    # bei mehreren set_value().run()-Zyklen auf demselben "at"-Objekt verwirrt.
    for option in ("sgd", "momentum", "adam"):
        at = _fresh()
        box = [s for s in at.selectbox if s.label == "Optimierer"][0]
        box.set_value(option).run()
        assert not at.exception
