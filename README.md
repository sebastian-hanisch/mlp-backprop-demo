# MLP + Backpropagation – wie eine verdeckte Schicht XOR löst – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-mlp-backprop-demo.streamlit.app/)**

Stück 2 der **Neuronale-Netze-Reihe** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Stück 1 zeigte: ein einzelnes Perceptron scheitert beweisbar am XOR-Muster (Minsky & Papert
1969). Eine **verdeckte Schicht** mit **Backpropagation** (Rumelhart, Hinton & Williams 1986)
löst genau das – und reduziert sich bei 0 verdeckten Einheiten **exakt** auf die
Perceptron-Regel aus Stück 1 zurück, keine Näherung.

**Einordnung in die Reihe:**

```
Perceptron (WURZEL)                              [gebaut]
 └─ MLP + Backpropagation                        [DIESES STÜCK]
      ├─ CNN                                     [gebaut]
      └─ RNN                                     [gebaut]
           └─ LSTM                               [gebaut]
                └─ Attention/Transformer         [gebaut]
```

**Ergebnis in Kürze:** 0 und 1 verdeckte Einheiten scheitern **strukturell** am XOR-Muster
(0/30 zufällige Initialisierungen), 2 Einheiten schaffen es bei 9/30 (30 %, nicht garantiert),
ab 3 fast immer. Bei 0 verdeckten Einheiten ist ein Gradientenschritt algebraisch **identisch**
zur Perceptron-Update-Regel (exakt gleiche Gewichtstrajektorie, geprüft). Eine von Hand
konstruierte 2-Einheiten-Lösung erreicht 0 Trainingsfehler ganz ohne Training. Überraschung beim
Optimierer-Vergleich: **Momentum konvergiert bei gleicher Lernrate schlechter als einfaches
SGD** (0/10 gegen 7/10) – ein konstant-großer Subgradient kombiniert mit hoher Massenträgheit
führt zu Oszillation statt Beschleunigung.

## Warum dieses Problem

Stück 1 endete mit der offenen Frage, woran ein einzelnes Perceptron scheitert und was der
nächste Schritt dagegen tut. Backpropagation (Rumelhart, Hinton & Williams 1986) ist die
Antwort: eine zusätzliche, nichtlineare Zwischenschicht, trainiert per Kettenregel. Dieses
Stück zeigt nicht nur, dass das funktioniert, sondern auch **warum** – über eine einheitliche
Verlustfunktion, die bei 0 verdeckten Einheiten beweisbar auf den Vorgänger zurückfällt.

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| 0 und 1 verdeckte Einheiten scheitern strukturell am XOR-Muster | ✅ 0/30 Erfolge bei beiden (`test_claim_hidden_unit_sweep_matches_readme_table`) |
| 2 verdeckte Einheiten lösen XOR bei einem Teil der Initialisierungen, nicht garantiert | ✅ 9/30 (30 %) |
| Ab 3 Einheiten gelingt es fast immer | ✅ 30/30, 29/30, 30/30, 30/30 (h=3,4,6,8) |
| Gradienten-Check gegen finite Differenzen unter $10^{-8}$ | ✅ 1,42·10⁻¹⁰ |
| MLP mit h=0 reduziert sich exakt auf die Perceptron-Regel | ✅ identische Gewichtstrajektorie (`test_claim_reduction_check_is_exact_match`) |
| Hand-konstruierte 2-Einheiten-Lösung löst XOR ohne Training | ✅ 100 % Trainingsgenauigkeit |
| ⚠️ Plan-Korrektur: erste Handkonstruktion (einzelne Achsen x1,x2 statt der Summe x1+x2) scheiterte | ⚠️ 0 % Genauigkeit – ein linearer Ausgang kann kein Produkt aus sign(x1),sign(x2) bilden; korrigiert auf zwei Einheiten, die dieselbe Summe x1+x2 mit verschobener Schwelle testen (OR-/AND-artig) |
| Adam/Momentum sind grundsätzlich schneller/zuverlässiger als SGD | ⚠️ **Nur teilweise:** Adam konvergiert zuverlässiger (9/10 gegen 7/10), aber Momentum bei gleicher Lernrate **schlechter** (0/10) – Oszillation statt Beschleunigung |

## Befunde (gemessen, keine Behauptungen)

**Hidden-Unit-Sweep** (XOR, 30 zufällige Initialisierungen je Größe, dieselben Daten):

| Verdeckte Einheiten h | Erfolge (von 30) | Erfolgsanteil |
|---|---|---|
| 0 | 0 | 0 % |
| 1 | 0 | 0 % |
| 2 | 9 | 30 % |
| 3 | 30 | 100 % |
| 4 | 29 | 97 % |
| 6 | 30 | 100 % |
| 8 | 30 | 100 % |

**Optimierer-Vergleich** (SGD/Momentum/Adam, gleiche Lernrate η=0,1, h=2, XOR, 10 Seeds):

| Optimierer | Konvergiert (von 10) | Ø Epochen (falls konvergiert) |
|---|---|---|
| SGD | 7 | 30,6 |
| Momentum | 0 | – |
| Adam | 9 | 38,0 |

**Korrektheits-Kette:** Perceptron (Stück 1) und MLP mit h=0 liefern auf denselben Daten
(gleiche Reihenfolge, gleiches η) exakt `w=[2,675, 0,429], b=-1,0` – identisch bis auf
Maschinengenauigkeit.

**Presets:** "0 verdeckte Einheiten = Perceptron" konvergiert in 2 Epochen (wie Stück 1), "2
verdeckte Einheiten löst XOR" (Seed 16) in 15 Epochen, "1 verdeckte Einheit scheitert" erreicht
nach 500 Epochen (Maximum) nie 0 Trainingsfehler.

## Modell und Verfahren

- `bp_scenario.py` – Datengeneratoren (identische Konstruktion wie in `perceptron-demo`,
  eigenständige Kopie ohne Cross-Repo-Import).
- `bp_mlp.py` – Kernalgorithmus: `MLP` (tanh-verdeckte Schicht + linearer Ausgang,
  Perceptron-Criterion-Verlust, Xavier-Init), `SGD`/`Momentum`/`Adam`.
- `bp_evaluation.py` – Hidden-Unit-Sweep, Gradienten-Check, Reduktions-Check, Optimierer-Vergleich.
- `bp_handcrafted.py` – die von Hand hergeleitete 2-Einheiten-XOR-Lösung.
- `bp_visualization.py` – Plotly: Kontur-Entscheidungsgrenze, Fehlerkurve, Sweep-Balken.

## Was die App zeigt

Datenmodus (trennbar/XOR), verdeckte Einheiten (0–8), Optimierer und die üblichen Regler in der
Sidebar; ein Epochen-Schieberegler zeigt die (jetzt nichtlineare) Entscheidungsgrenze und die
Fehlerkurve Schritt für Schritt; der Hidden-Unit-Sweep, die Hand-Lösung, die Korrektheits-Kette
zum Perceptron und ein "📐"-Abschnitt (Gradienten-Check, Optimierer-Tabelle) folgen darunter.

## Was nicht funktioniert hat / Grenzen

**Echter Fehler in der Vormessung gefunden und behoben:** Die erste Handkonstruktion der
XOR-Lösung testete `x1` und `x2` in getrennten verdeckten Einheiten (in der Annahme, ein
linearer Ausgang könnte `sign(x1)·sign(x2)` nachbilden) – das schlug mit 0 % Genauigkeit fehl,
weil ein linearer Ausgang Hidden-Aktivierungen nur **addieren**, nicht **multiplizieren** kann.
Korrigiert auf zwei Einheiten, die dieselbe Summe `x1+x2` mit verschobener Schwelle testen
(OR-/AND-artig); die beiden "gemischten" Quadranten fallen im verdeckten Raum auf denselben
Punkt zusammen, was hier unschädlich ist, da sie dasselbe Ziel-Label teilen. Siehe
`bp_handcrafted.py` für die vollständige Herleitung.

**Grenzen:** Eine verdeckte Schicht, kleine synthetische 2D-Vehikel, keine echten Datensätze.
Der Optimierer-Vergleich nutzt für alle drei Optimierer dieselbe Lernrate – ein separat
getuntes Momentum könnte besser abschneiden; die Demo zeigt bewusst den "naiven Vergleich ohne
Nachjustieren", weil genau das eine reale Falle ist.

## Tests

39 Tests, `python -m pytest tests/ -v`:
- `test_scenario.py` – Reproduzierbarkeit, Trennbarkeitsgarantie, XOR-Struktur.
- `test_mlp.py` – Forward/Backward, Gradienten-Check (h=0 und h>0), Hidden-Unit-Grenzfälle,
  Optimierer-Verbesserung über mehrere Seeds.
- `test_handcrafted.py` – Hand-Lösung erreicht 100 % über mehrere Seeds.
- `test_evaluation.py` – Sweep, Reduktions-Check, Optimierer-Vergleich.
- `test_presets.py`, `test_claims.py` – jede Zahl oben nachgerechnet.
- `test_app.py` – Streamlit `AppTest`: Presets, Regler-Extremwerte, Optimierer-Wechsel, Footer.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `bp_constants.py` | Regler-Grenzen, Presets |
| `bp_scenario.py` | Datengeneratoren |
| `bp_mlp.py` | Netz, Optimierer |
| `bp_evaluation.py` | Sweep, Gradienten-Check, Reduktions-Check |
| `bp_handcrafted.py` | Hand-Lösung |
| `bp_visualization.py` | Plotly-Plots |
| `bp_presets.py` | Permalink-Sync, Presets |
| `tests/` | pytest-Suite |

## Bewusst nicht umgesetzt

Kein Nachjustieren der Optimierer-Lernraten (siehe Grenzen oben) – bewusst, um den naiven
Vergleich als eigenständigen Befund zu zeigen. Keine tieferen Netze (mehr als eine verdeckte
Schicht) – das würde die klare "eine Schicht löst XOR"-Botschaft dieses Stücks verwässern.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
streamlit run app.py
```

## Literatur

- Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986). *Learning representations by
  back-propagating errors.* Nature, 323(6088), 533–536.
- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and
  Organization in the Brain.* Psychological Review, 65(6), 386–408. (Perceptron-Criterion-Verlust)
- Minsky, M. & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry.*
  MIT Press. (XOR-Grenze des Vorgängers)

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Neuronale Netze: vom Perceptron zum Transformer](https://sebastianhanisch.net/konzepte-neuronale-netze.html).
