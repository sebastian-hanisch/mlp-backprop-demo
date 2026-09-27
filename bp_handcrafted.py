"""Die von Hand hergeleitete XOR-Loesung mit 2 verdeckten Einheiten - keine
einzige Trainings-Iteration, nur eine gezielt konstruierte Gewichtswahl.

Herleitung: normierte Eingaben liegen nahe (+-1,+-1) (vier Quadranten). Eine
einzelne verdeckte Einheit kann kein Produkt aus sign(x1) und sign(x2) bilden
(ein linearer Ausgang kann Hidden-Aktivierungen nur ADDIEREN, nicht
multiplizieren) - daher testen BEIDE Einheiten dieselbe Summe x1+x2 mit
verschobener Schwelle:
  h1 = tanh(gain*(x1+x2+1))  ~ OR(x1>0, x2>0)   (-1 nur im Quadranten --)
  h2 = tanh(gain*(x1+x2-1))  ~ AND(x1>0, x2>0)  (+1 nur im Quadranten ++)
Die beiden Quadranten +- und -+ liegen BEIDE bei x1+x2=0 und fallen auf
denselben Punkt (h1,h2)=(+1,-1) - unschaedlich, da sie dasselbe Ziel-Label
(-1) teilen. Die drei verbleibenden Punkte in h-Raum -
(+1,+1)->+1, (-1,-1)->+1, (+1,-1)->-1 - sind linear trennbar durch
Ausgang = -h1 + h2 + 1."""
import numpy as np

import bp_mlp as mlp_mod

GAIN = 8.0


def build_handcrafted_xor_net() -> mlp_mod.MLP:
    net = mlp_mod.MLP(2, 2, seed=0)  # seed irrelevant, Gewichte werden ueberschrieben
    net.params["W1"] = GAIN * np.array([[1.0, 1.0], [1.0, 1.0]])
    net.params["b1"] = GAIN * np.array([1.0, -1.0])
    net.params["w2"] = np.array([-1.0, 1.0])
    net.params["b2"] = np.asarray(1.0)
    return net
