"""
Plotly visualisation helpers for Solar PV fault diagnosis (dark solar theme).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Theme ─────────────────────────────────────────────────────────────────────
_BG = "#0f0f1a"
_PAPER = "#1a1a2e"
_GRID = "#2a2a4a"
_TEXT = "#e0e0ff"
_ACCENT = "#f39c12"

_FAULT_COLORS = {
    "Normal": "#2ecc71",
    "Partial Shading": "#3498db",
    "Soiling": "#f39c12",
    "Hot Spot": "#e74c3c",
    "PID Effect": "#9b59b6",
    "Bypass Diode Failure": "#e91e63",
    "String Disconnect": "#1abc9c",
}

_BASE_LAYOUT = dict(
    plot_bgcolor=_BG,
    paper_bgcolor=_PAPER,
    font=dict(color=_TEXT, family="Inter, sans-serif"),
    xaxis=dict(gridcolor=_GRID, zeroline=False),
    yaxis=dict(gridcolor=_GRID, zeroline=False),
    margin=dict(l=50, r=30, t=50, b=50),
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_BASE_LAYOUT)
    return fig


# ── Fault timeline ─────────────────────────────────────────────────────────────

def plot_fault_timeline(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of fault labels over time."""
    fig = go.Figure()
    ts_col = "timestamp" if "timestamp" in df.columns else df.index.name or None

    fault_col = df["fault_label"] if "fault_label" in df.columns else pd.Series(
        [0] * len(df))
    label_names = df["fault_name"] if "fault_name" in df.columns else fault_col.astype(str)
    x_vals = df[ts_col] if ts_col and ts_col in df.columns else df.index

    for label_val in sorted(fault_col.unique()):
        mask = fault_col == label_val
        name = label_names[mask].iloc[0] if mask.any() else str(label_val)
        color = _FAULT_COLORS.get(name, _ACCENT)
        fig.add_trace(go.Scatter(
            x=x_vals[mask], y=[label_val] * mask.sum(),
            mode="markers", marker=dict(color=color, size=5),
            name=name, legendgroup=name,
        ))

    fig.update_layout(title="Fault Timeline", xaxis_title="Time",
                      yaxis_title="Fault Label", **_BASE_LAYOUT)
    return fig


# ── Fault heatmap ─────────────────────────────────────────────────────────────

def plot_fault_heatmap(df: pd.DataFrame) -> go.Figure:
    """Hour-of-day vs day-of-week fault frequency heatmap."""
    if "timestamp" not in df.columns:
        return go.Figure().update_layout(**_BASE_LAYOUT)

    tmp = df.copy()
    tmp["hour"] = pd.to_datetime(tmp["timestamp"]).dt.hour
    tmp["dow"] = pd.to_datetime(tmp["timestamp"]).dt.day_name()
    tmp["is_fault"] = (tmp["fault_label"] != 0).astype(int)

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = tmp.groupby(["dow", "hour"])["is_fault"].mean().unstack(fill_value=0)
    pivot = pivot.reindex([d for d in days if d in pivot.index])

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(range(24)),
        y=pivot.index.tolist(),
        colorscale="YlOrRd",
        showscale=True,
    ))
    fig.update_layout(title="Fault Frequency: Hour × Day of Week",
                      xaxis_title="Hour of Day", yaxis_title="Day",
                      **_BASE_LAYOUT)
    return fig


# ── System health gauge ────────────────────────────────────────────────────────

def plot_system_health_gauge(health_score: float) -> go.Figure:
    color = "#2ecc71" if health_score > 75 else ("#f39c12" if health_score > 50 else "#e74c3c")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        delta={"reference": 100, "suffix": "%"},
        title={"text": "System Health Score", "font": {"color": _TEXT}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": _TEXT},
            "bar": {"color": color},
            "bgcolor": _BG,
            "bordercolor": _GRID,
            "steps": [
                {"range": [0, 50], "color": "#3d0000"},
                {"range": [50, 75], "color": "#3d2200"},
                {"range": [75, 100], "color": "#003d00"},
            ],
            "threshold": {"line": {"color": _ACCENT, "width": 4}, "value": 90},
        },
        number={"suffix": "%", "font": {"color": _TEXT}},
    ))
    fig.update_layout(paper_bgcolor=_PAPER, font=dict(color=_TEXT),
                      margin=dict(l=30, r=30, t=60, b=30))
    return fig


# ── Energy loss bars ──────────────────────────────────────────────────────────

def plot_energy_loss_bars(energy_loss: dict) -> go.Figure:
    names = list(energy_loss.keys())
    values = [energy_loss[n] for n in names]
    colors = [_FAULT_COLORS.get(n, _ACCENT) for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors,
        text=[f"{v:.2f} kWh" for v in values],
        textposition="outside",
    ))
    fig.update_layout(title="Estimated Energy Loss by Fault Type",
                      xaxis_title="Fault Type", yaxis_title="Energy Lost (kWh)",
                      **_BASE_LAYOUT)
    return fig


# ── Sensor grid ───────────────────────────────────────────────────────────────

def plot_sensor_grid(df: pd.DataFrame, feature_cols: list | None = None) -> go.Figure:
    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if c not in ("timestamp", "fault_label", "fault_name")][:9]
    ts_col = "timestamp" if "timestamp" in df.columns else None
    x_vals = df[ts_col] if ts_col else df.index
    rows, cols = 3, 3
    titles = feature_cols[:rows * cols]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

    palette = px.colors.qualitative.Plotly
    for idx, col in enumerate(titles):
        r, c = divmod(idx, cols)
        fig.add_trace(
            go.Scatter(x=x_vals, y=df[col], mode="lines",
                       line=dict(color=palette[idx % len(palette)], width=1),
                       name=col, showlegend=False),
            row=r + 1, col=c + 1,
        )

    fig.update_layout(height=600, title_text="Sensor Overview",
                      plot_bgcolor=_BG, paper_bgcolor=_PAPER,
                      font=dict(color=_TEXT),
                      margin=dict(l=40, r=20, t=60, b=40))
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor=_GRID, zeroline=False)
    return fig


# ── Causal graph ──────────────────────────────────────────────────────────────

def plot_causal_graph(G, physics_validation: list | None = None) -> go.Figure:
    """Render a NetworkX DiGraph as an interactive Plotly network."""
    import networkx as nx

    if len(G.nodes) == 0:
        return go.Figure().update_layout(**_BASE_LAYOUT)

    valid_edges: set = set()
    if physics_validation:
        valid_edges = {r["relation"] for r in physics_validation if r["found"]}

    pos = nx.spring_layout(G, seed=42, k=2)

    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        color = "#2ecc71" if (u, v) in valid_edges else "#3498db"
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(1, data.get("strength", 0.5) * 4), color=color),
            hoverinfo="none", showlegend=False,
        ))

    node_x = [pos[n][0] for n in G.nodes]
    node_y = [pos[n][1] for n in G.nodes]
    degree = dict(G.degree())
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=[10 + degree[n] * 3 for n in G.nodes],
                    color=_ACCENT, line=dict(width=2, color=_BG)),
        text=list(G.nodes), textposition="top center",
        hoverinfo="text",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    layout = dict(_BASE_LAYOUT)
    layout['xaxis'] = dict(visible=False)
    layout['yaxis'] = dict(visible=False)
    fig.update_layout(title="Causal Graph", showlegend=False, **layout)
    return fig


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(history: dict) -> go.Figure:
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss", "Accuracy"])
    for name, values in [("Train Loss", history.get("train_loss", [])),
                          ("Val Loss", history.get("val_loss", []))]:
        fig.add_trace(go.Scatter(x=epochs, y=values, name=name, mode="lines"),
                      row=1, col=1)
    for name, values in [("Train Acc", history.get("train_acc", [])),
                          ("Val Acc", history.get("val_acc", []))]:
        fig.add_trace(go.Scatter(x=epochs, y=values, name=name, mode="lines"),
                      row=1, col=2)
    fig.update_layout(title="Training History", plot_bgcolor=_BG,
                      paper_bgcolor=_PAPER, font=dict(color=_TEXT),
                      margin=dict(l=40, r=20, t=60, b=40))
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor=_GRID, zeroline=False)
    return fig


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, fault_labels: dict) -> go.Figure:
    labels = [fault_labels.get(i, str(i)) for i in range(len(fault_labels))]
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    text = [[f"{cm[i][j]}<br>({cm_norm[i][j]:.1%})"
             for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(
        z=cm_norm, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale="Blues", showscale=True,
    ))
    fig.update_layout(title="Confusion Matrix",
                      xaxis_title="Predicted", yaxis_title="Actual",
                      **_BASE_LAYOUT)
    return fig


# ── Feature attribution ───────────────────────────────────────────────────────

def plot_feature_attribution(attribution: dict) -> go.Figure:
    items = sorted(attribution.items(), key=lambda kv: kv[1])
    features, scores = zip(*items) if items else ([], [])
    fig = go.Figure(go.Bar(
        x=list(scores), y=list(features),
        orientation="h",
        marker_color=[_ACCENT if s > np.mean(scores) else "#3498db"
                      for s in scores],
    ))
    fig.update_layout(title="Feature Attribution (Input × Gradient)",
                      xaxis_title="Importance", **_BASE_LAYOUT)
    return fig


# ── Causal chain ──────────────────────────────────────────────────────────────

def plot_causal_chain(chain: list) -> go.Figure:
    """Sankey diagram for a list of (source, target, value) tuples or edge dicts."""
    if not chain:
        return go.Figure().update_layout(**_BASE_LAYOUT)

    nodes: list[str] = []
    links: dict = {"source": [], "target": [], "value": []}
    for item in chain:
        if isinstance(item, dict):
            src, tgt, val = item.get("cause", "?"), item.get("effect", "?"), item.get("strength", 1.0)
        else:
            src, tgt, val = item[0], item[1], (item[2] if len(item) > 2 else 1.0)
        for n in (src, tgt):
            if n not in nodes:
                nodes.append(n)
        links["source"].append(nodes.index(src))
        links["target"].append(nodes.index(tgt))
        links["value"].append(float(val))

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, color=_ACCENT, pad=15, thickness=20),
        link=dict(**links, color="rgba(243,156,18,0.4)"),
    ))
    fig.update_layout(title="Causal Chain", paper_bgcolor=_PAPER,
                      font=dict(color=_TEXT), margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ── Model comparison ──────────────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame) -> go.Figure:
    metrics = ["Accuracy", "F1 Macro", "Precision", "Recall"]
    metrics = [m for m in metrics if m in results_df.columns]
    fig = go.Figure()
    palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for idx, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=results_df["Model"],
            y=results_df[metric],
            marker_color=palette[idx % len(palette)],
        ))
    fig.update_layout(barmode="group", title="Model Comparison",
                      yaxis_title="Score", **_BASE_LAYOUT)
    return fig


# ── Probability bars ──────────────────────────────────────────────────────────

def plot_probability_bars(probabilities: dict) -> go.Figure:
    names = list(probabilities.keys())
    values = [probabilities[n] for n in names]
    colors = [_FAULT_COLORS.get(n, _ACCENT) for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(title="Fault Probability Distribution",
                      xaxis_title="Fault Type", yaxis_title="Probability",
                      yaxis_range=[0, 1.05], **_BASE_LAYOUT)
    return fig
