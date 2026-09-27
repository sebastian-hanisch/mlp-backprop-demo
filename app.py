"""MLP + Backpropagation — wie eine verdeckte Schicht XOR löst

Sebastian Hanisch - Operations Research und Machine Learning

Stück 2 der "Neuronale Netze"-Reihe der "Konzepte"-Reihe:
Perceptron -> MLP+Backpropagation -> {CNN, RNN -> LSTM -> Attention/Transformer}.
Ein einzelnes Perceptron (Stück 1) scheitert beweisbar am XOR-Muster. Eine
verdeckte Schicht mit Backpropagation (Rumelhart, Hinton & Williams 1986) löst
genau das - und reduziert sich bei 0 verdeckten Einheiten exakt auf den
Vorgänger zurück.

Lauffähig mit: streamlit run app.py
"""
import numpy as np
import streamlit as st

import bp_constants as C
import bp_evaluation as ev
import bp_handcrafted as hc
import bp_mlp as mlp_mod
import bp_presets as pr
import bp_visualization as viz

st.set_page_config(page_title="MLP + Backpropagation", layout="wide")


@st.cache_data(show_spinner=False)
def _analyse(mode, n, seed, eta, hidden, max_epochs, optimizer, gap):
    settings = ev.Settings(mode=mode, n=n, seed=seed, eta=eta, hidden=hidden,
                           max_epochs=max_epochs, optimizer=optimizer, gap=gap)
    out = ev.analyse(settings)
    return out["scenario"], out["result"], out["net"]


@st.cache_data(show_spinner=False)
def _hidden_sweep():
    return ev.hidden_unit_sweep()


@st.cache_data(show_spinner=False)
def _gradient_check():
    return ev.gradient_check()


@st.cache_data(show_spinner=False)
def _reduction_check():
    out = ev.reduction_check()
    return out["identical"], out["w_perceptron"].tolist(), out["b_perceptron"], \
        out["w_mlp"].tolist(), out["b_mlp"]


@st.cache_data(show_spinner=False)
def _optimizer_comparison():
    return ev.optimizer_comparison()


@st.cache_data(show_spinner=False)
def _handcrafted_accuracy():
    from bp_scenario import make_xor
    net = hc.build_handcrafted_xor_net()
    X, y = make_xor(n=40, seed=1, spread=C.SPREAD_XOR, gap=C.GAP_XOR_DEFAULT)
    X_norm = X / C.GAP_XOR_DEFAULT
    preds = net.predict(X_norm)
    return float(np.mean(preds == y))


st.title("🧠 MLP + Backpropagation — wie eine verdeckte Schicht XOR löst")
st.markdown(
    "Stück 1 zeigte: ein einzelnes Perceptron scheitert beweisbar am XOR-Muster. Eine "
    "**verdeckte Schicht** mit **Backpropagation** löst genau das - und reduziert sich bei "
    "0 verdeckten Einheiten exakt auf die Perceptron-Regel zurück (keine Näherung: dieselbe "
    "Update-Formel, dieselbe Trajektorie)."
)
st.caption(
    "Stück 2 (Wurzel: Perceptron) der 'Neuronale Netze'-Reihe. Geplante Folgestücke "
    "(noch nicht gebaut): CNN, RNN, LSTM, Attention/Transformer."
)

with st.expander("So funktioniert das MLP", expanded=True):
    st.markdown(
        "1. Verdeckte Schicht: $h = \\tanh(W_1 x + b_1)$ ($h$ verdeckte Einheiten).\n"
        "2. Linearer Ausgang: $s = w_2 \\cdot h + b_2$ (bei $h=0$ direkt $s = w \\cdot x + b$).\n"
        "3. Perceptron-Criterion-Verlust: $L = \\max(0, -y\\,s)$ - derselbe Verlust wie in "
        "Stück 1, hier nur auf eine (ggf. nichtlineare) Merkmalstransformation angewandt.\n"
        "4. Backpropagation (Kettenregel) berechnet den Gradienten durch die verdeckte "
        "Schicht; ein Optimierer (SGD/Momentum/Adam) aktualisiert die Gewichte."
    )

st.caption("🎯 Schnellstart – ein Klick lädt ein durchgerechnetes Beispiel:")
preset_cols = st.columns(len(C.PRESETS))
for col, (key, preset) in zip(preset_cols, C.PRESETS.items()):
    with col:
        st.button(preset["label"], help=preset["help"], on_click=pr.apply_preset, args=(key,),
                   use_container_width=True)

st.caption("🔗 Die Adresszeile speichert deine Einstellungen als Permalink.")

pr.load_permalink_settings()
pr.init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    mode = st.radio("Datenmodus", C.MODES, format_func=lambda m: C.MODE_LABELS[m],
                    key="widget_mode", index=C.MODES.index(ss["mode"]),
                    on_change=pr.store_from_widget, args=("mode",))
    ss["mode"] = mode
    b = pr.bounds(mode)
    gap_value = min(max(ss["gap"], b["gap_min"]), b["gap_max"])
    gap = st.slider("Abstand der Zentren (gap)", b["gap_min"], b["gap_max"], gap_value, step=0.05,
                    key="widget_gap", on_change=pr.store_from_widget, args=("gap",))
    ss["gap"] = gap
    hidden = st.slider("Verdeckte Einheiten h", C.HIDDEN_MIN, C.HIDDEN_MAX, ss["hidden"],
                       key="widget_hidden", on_change=pr.store_from_widget, args=("hidden",))
    ss["hidden"] = hidden
    optimizer = st.selectbox("Optimierer", C.OPTIMIZERS, format_func=lambda o: C.OPTIMIZER_LABELS[o],
                             index=C.OPTIMIZERS.index(ss["optimizer"]), key="widget_optimizer",
                             on_change=pr.store_from_widget, args=("optimizer",))
    ss["optimizer"] = optimizer
    n = st.slider("Anzahl Punkte n", C.N_MIN, C.N_MAX, ss["n"], step=4,
                  key="widget_n", on_change=pr.store_from_widget, args=("n",))
    ss["n"] = n
    eta = st.slider("Lernrate η", C.ETA_MIN, C.ETA_MAX, ss["eta"], step=0.05,
                    key="widget_eta", on_change=pr.store_from_widget, args=("eta",))
    ss["eta"] = eta
    max_epochs = st.slider("Max. Epochen", C.MAX_EPOCHS_MIN, C.MAX_EPOCHS_MAX, ss["max_epochs"],
                           step=50, key="widget_max_epochs",
                           on_change=pr.store_from_widget, args=("max_epochs",))
    ss["max_epochs"] = max_epochs
    seed = st.number_input("Seed", value=ss["seed"], step=1,
                           key="widget_seed", on_change=pr.store_from_widget, args=("seed",))
    ss["seed"] = seed
    st.button("🎲 Zufälliger Seed", on_click=pr.randomize_seed)

pr.sync_query_params(dict(mode=mode, n=n, seed=seed, eta=eta, hidden=hidden,
                         max_epochs=max_epochs, optimizer=optimizer, gap=gap))

with st.spinner("Rechne..."):
    scenario, result, net = _analyse(mode, n, int(seed), eta, hidden, max_epochs, optimizer, gap)

st.markdown("---")
st.subheader("🎯 Konvergenz Schritt für Schritt")
max_step = len(result.errors_per_epoch)
step = st.select_slider("Epoche", options=list(range(1, max_step + 1)), value=max_step,
                        key=f"step_{mode}_{n}_{seed}_{gap}_{hidden}_{optimizer}_{max_epochs}")

# Netz-Zustand bei "step" neu trainieren (deterministisch, gleicher Seed/Optimierer)
net_at_step = mlp_mod.MLP(2, hidden, seed=int(seed))
opt_at_step = mlp_mod.make_optimizer(optimizer, eta)
for _ in range(step):
    mlp_mod.train_epoch(net_at_step, opt_at_step, scenario.X, scenario.y)

x_range, y_range = viz.data_bounds(scenario.X)
col_left, col_right = st.columns([3, 2])
with col_left:
    fig_boundary = viz.build_boundary_figure(scenario.X, scenario.y, net_at_step, x_range, y_range,
                                             title=f"Entscheidungsgrenze nach Epoche {step}")
    st.plotly_chart(fig_boundary, key=f"boundary_{mode}_{step}_{n}_{seed}_{gap}_{hidden}_{optimizer}",
                    use_container_width=True)
with col_right:
    fig_errors = viz.build_errors_figure(result.errors_per_epoch, current_epoch=step)
    st.plotly_chart(fig_errors, key=f"errors_{mode}_{step}_{n}_{seed}_{gap}_{hidden}_{optimizer}",
                    use_container_width=True)

st.markdown("---")
st.subheader("🎯 Was am Ende steht")
m1, m2, m3 = st.columns(3)
m1.metric("Konvergiert?", "Ja" if result.converged else "Nein (Max. Epochen erreicht)")
m2.metric("Epochen bis zum Ende", f"{result.epochs}")
m3.metric("Verdeckte Einheiten", f"{hidden}")

st.markdown("---")
st.subheader("🎯 Wie viele verdeckte Einheiten braucht XOR mindestens?")
sweep = _hidden_sweep()
st.plotly_chart(viz.build_hidden_sweep_figure(sweep), key="hidden_sweep_chart",
                use_container_width=True)
st.caption(
    "0 und 1 verdeckte Einheiten scheitern **strukturell** (0 % Erfolg über 30 zufällige "
    "Initialisierungen) - wie das Perceptron. Bei 2 Einheiten gelingt es bei einem Teil der "
    "Initialisierungen (nicht garantiert), ab 3 fast immer."
)

st.subheader("🎯 Hand-konstruierte XOR-Lösung (ohne jedes Training)")
hc_acc = _handcrafted_accuracy()
st.metric("Trainingsgenauigkeit der von Hand hergeleiteten Gewichte", f"{hc_acc*100:.0f}%")
st.caption(
    "Zwei verdeckte Einheiten testen dieselbe Summe x1+x2 mit verschobener Schwelle "
    "(OR- bzw. AND-artig); ein linearer Ausgang trennt die drei verbleibenden Punkte im "
    "verdeckten Raum. Details im 📐-Abschnitt."
)

with st.expander("🔬 Korrektheits-Kette: reduziert sich das MLP auf das Perceptron?"):
    identical, w_p, b_p, w_m, b_m = _reduction_check()
    st.markdown(
        f"Bei $h=0$ mit demselben Perceptron-Criterion-Verlust und SGD ist ein "
        f"Gradientenschritt bei Fehlklassifikation algebraisch $w \\leftarrow w + \\eta y x$ - "
        f"**identisch** zur Perceptron-Regel aus Stück 1, keine Näherung."
    )
    c1, c2 = st.columns(2)
    c1.metric("Perceptron (Stück 1)", f"w={np.round(w_p, 3)}, b={round(b_p, 3)}")
    c2.metric("MLP mit h=0", f"w={np.round(w_m, 3)}, b={round(b_m, 3)}")
    st.metric("Trajektorien identisch?", "Ja" if identical else "Nein (Fehler!)")

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n"
    "|---|---|---|\n"
    "| Eine verdeckte Schicht reicht | Bei komplexeren Mustern (z. B. Bildern) braucht es "
    "räumliche Struktur statt nur mehr Einheiten | CNN (nächstes Stück) |\n"
    "| Gradienten bleiben gut skaliert | Bei tiefen/rekurrenten Netzen können sie verschwinden "
    "oder explodieren | RNN/LSTM (spätere Stücke) |\n"
    "| Backprop findet eine Lösung | Bei $h=2$ gelingt das nur bei einem Teil der "
    "Initialisierungen (lokale Minima) | — |\n"
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Vorwärtsdurchlauf:** $h = \tanh(W_1 x + b_1)$, $s = w_2 \cdot h + b_2$ (bei $h=0$: $s=w\cdot x+b$).

**Verlust:** $L = \max(0, -y s)$ (Perceptron-Criterion, Rosenblatt 1958 in Verlustform).

**Rückwärtsdurchlauf** (Kettenregel, bei $ys \le 0$): $\partial L/\partial s = -y$,
$\partial L/\partial w_2 = -y\,h$, $\partial L/\partial h = -y\,w_2$,
$\partial L/\partial z_1 = (-y\,w_2) \odot (1-h^2)$ ($\tanh'(z)=1-\tanh^2(z)$),
$\partial L/\partial W_1 = (\partial L/\partial z_1)\, x^\top$.

**Reduktion auf das Perceptron** ($h=0$): $s=w\cdot x+b$, Gradient bei $ys\le 0$ ist
$\partial L/\partial w = -yx$ — ein SGD-Schritt $w \leftarrow w-\eta(-yx)=w+\eta y x$ ist
exakt die Perceptron-Regel aus Stück 1, keine Näherung.

**Gradienten-Check:** maximaler relativer Fehler gegen finite Differenzen (aktuell gemessen):
"""
    )
    st.metric("Maximaler relativer Fehler", f"{_gradient_check():.2e}")
    st.markdown("**Optimierer-Vergleich** (SGD/Momentum/Adam, $h=2$, XOR, 10 Seeds):")
    opt_rows = _optimizer_comparison()
    st.dataframe(
        {"Optimierer": [C.OPTIMIZER_LABELS[r["optimizer"]] for r in opt_rows],
         "Konvergiert (von 10)": [r["n_converged"] for r in opt_rows],
         "Ø Epochen (falls konvergiert)": [f"{r['mean_epochs']:.1f}" if r["mean_epochs"] else "–"
                                           for r in opt_rows]},
        hide_index=True, use_container_width=True,
    )
    st.markdown(
        "**Literatur:** Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986). "
        "*Learning representations by back-propagating errors.* Nature, 323(6088), 533–536."
    )
    st.caption(
        "Implementiert in `bp_mlp.py` (Netz, Optimierer), `bp_scenario.py` (Daten), "
        "`bp_evaluation.py` (Sweep, Gradienten-Check, Reduktions-Check), "
        "`bp_handcrafted.py` (Hand-Lösung), `bp_visualization.py` (Plots)."
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) "
    "– Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung "
    "für Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
