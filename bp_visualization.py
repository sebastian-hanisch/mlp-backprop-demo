"""Reine Plotly-Figure-Builder, keine Streamlit-Aufrufe. Achsen werden fest
uebergeben (feste Wertebereiche, feedback_plotly_fixedrange_convention)."""
import numpy as np
import plotly.graph_objects as go

COLOR_POS = "#1f77b4"
COLOR_NEG = "#d62728"
COLORSCALE = [[0.0, "#f9d0cd"], [0.5, "#ffffff"], [1.0, "#c9e0f5"]]


def data_bounds(X: np.ndarray, pad: float = 1.0):
    x_min, y_min = X.min(axis=0) - pad
    x_max, y_max = X.max(axis=0) + pad
    return (float(x_min), float(x_max)), (float(y_min), float(y_max))


def build_boundary_figure(X, y, net, x_range, y_range, title="", resolution=50):
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    scores = np.zeros((resolution, resolution))
    for i, yy in enumerate(ys):
        for j, xx in enumerate(xs):
            s, _ = net.forward(np.array([xx, yy]))
            scores[i, j] = s
    bound = np.max(np.abs(scores)) or 1.0

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=scores, colorscale=COLORSCALE, zmin=-bound, zmax=bound,
        contours=dict(start=-bound, end=bound, size=bound / 10),
        showscale=False, line=dict(width=0), opacity=0.85, name="Score",
    ))
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=scores, showscale=False, contours=dict(coloring="lines", start=0, end=0, size=1),
        line=dict(color="#2ca02c", width=3), name="Entscheidungsgrenze (Score=0)",
    ))
    pos = X[y > 0]
    neg = X[y < 0]
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers", name="Klasse +1",
                             marker=dict(color=COLOR_POS, size=8, line=dict(width=1, color="white"))))
    fig.add_trace(go.Scatter(x=neg[:, 0], y=neg[:, 1], mode="markers", name="Klasse -1",
                             marker=dict(color=COLOR_NEG, size=8, line=dict(width=1, color="white"))))
    fig.update_layout(
        title=title, xaxis=dict(range=list(x_range), fixedrange=True, title="x1"),
        yaxis=dict(range=list(y_range), fixedrange=True, title="x2", scaleanchor="x"),
        showlegend=True, height=420, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_errors_figure(errors_per_epoch, current_epoch=None, title="Fehlerzahl je Epoche"):
    epochs = list(range(1, len(errors_per_epoch) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=errors_per_epoch, mode="lines", name="Fehler",
                             line=dict(color=COLOR_NEG, width=2)))
    if current_epoch is not None and 1 <= current_epoch <= len(errors_per_epoch):
        fig.add_trace(go.Scatter(x=[current_epoch], y=[errors_per_epoch[current_epoch - 1]],
                                 mode="markers", name="aktuelle Epoche",
                                 marker=dict(color="#2ca02c", size=12, symbol="star")))
    fig.update_layout(
        title=title, xaxis=dict(title="Epoche", fixedrange=True),
        yaxis=dict(title="Anzahl Fehler", fixedrange=True, rangemode="tozero"),
        showlegend=False, height=300, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_hidden_sweep_figure(rows, title="Erfolgsanteil je Anzahl verdeckter Einheiten"):
    hiddens = [r["hidden"] for r in rows]
    rates = [r["rate"] * 100 for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(h) for h in hiddens], y=rates, marker_color=COLOR_POS))
    fig.update_layout(
        title=title, xaxis=dict(title="Verdeckte Einheiten h", fixedrange=True),
        yaxis=dict(title="Erfolgsanteil (%)", fixedrange=True, range=[0, 105]),
        height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
