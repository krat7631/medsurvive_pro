"""Plotly figures with one consistent look."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .analysis import KMCurve

PALETTE = ["#93B4FF", "#F2A65A", "#9B91E8", "#7FD3A8", "#FF8A7A", "#E3D58A"]
INK, MUTED, GRID, BG = "#F4F3F0", "#A3A8B4", "#2A3242", "rgba(0,0,0,0)"


def _layout(fig: go.Figure, height: int = 420, **kwargs) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=INK, size=13),
        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08, x=0),
        hoverlabel=dict(bgcolor="#171C27", font_color=INK), **kwargs,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def km_figure(curves: list[KMCurve], time_unit: str) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(curves):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(x=c.timeline + c.timeline[::-1], y=c.upper + c.lower[::-1], fill="toself",
                                 fillcolor=_rgba(color, 0.12), line=dict(width=0), hoverinfo="skip",
                                 showlegend=False, line_shape="hv"))
        label = f"{c.group} (n={c.n})"
        fig.add_trace(go.Scatter(x=c.timeline, y=c.survival, mode="lines", line=dict(color=color, width=2.2),
                                 line_shape="hv", name=label,
                                 hovertemplate=f"{c.group}<br>%{{x:.0f}} {time_unit}: %{{y:.1%}}<extra></extra>"))
        if c.censor_times:
            surv_at = np.interp(c.censor_times, c.timeline, c.survival)
            fig.add_trace(go.Scatter(x=c.censor_times, y=surv_at, mode="markers", showlegend=False,
                                     marker=dict(symbol="line-ns", size=7, line=dict(width=1, color=color)),
                                     hovertemplate=f"censored at %{{x:.0f}} {time_unit}<extra></extra>"))
    fig.update_yaxes(title="Survival probability", tickformat=".0%", range=[0, 1.02])
    fig.update_xaxes(title=f"Time ({time_unit})")
    return _layout(fig)


def forest_figure(table: pd.DataFrame, title: str = "") -> go.Figure:
    """Hazard ratios with 95% CIs on a log axis; 1.0 = no difference in risk."""
    t = table.iloc[::-1]
    colors = [PALETTE[1] if lo > 1 else PALETTE[0] if hi < 1 else MUTED
              for lo, hi in zip(t["ci_lower"], t["ci_upper"], strict=True)]
    fig = go.Figure(go.Scatter(
        x=t["hazard_ratio"], y=t.index, mode="markers", marker=dict(size=10, color=colors),
        error_x=dict(type="data", symmetric=False, array=t["ci_upper"] - t["hazard_ratio"],
                     arrayminus=t["hazard_ratio"] - t["ci_lower"], color=MUTED, thickness=1.5),
        customdata=np.stack([t["ci_lower"], t["ci_upper"], t["p_value"]], axis=1),
        hovertemplate="%{y}<br>HR %{x:.2f} [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
                      "<br>p = %{customdata[2]:.3g}<extra></extra>",
    ))
    fig.add_vline(x=1, line=dict(color=MUTED, dash="dash", width=1))
    fig.update_xaxes(type="log", title="Hazard ratio (log scale, 95% CI)")
    return _layout(fig, height=max(220, 46 * len(t) + 90), title=title, showlegend=False)


def cv_figure(cv: dict, names: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    metrics = [("harrell_c", "Harrell's C"), ("uno_c", "Uno's C")]
    for i, (key, label) in enumerate(metrics):
        means = [cv["models"][m][key]["mean"] for m in names]
        stds = [cv["models"][m][key]["std"] for m in names]
        fig.add_trace(go.Bar(x=list(names.values()), y=means, name=label, marker_color=PALETTE[i],
                             error_y=dict(type="data", array=stds, color=MUTED),
                             hovertemplate="%{x}<br>" + label + " %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.5, line=dict(color=MUTED, dash="dash", width=1),
                  annotation_text="chance", annotation_font_color=MUTED)
    fig.update_yaxes(title="C-index (higher is better)", range=[0.4, 0.85])
    return _layout(fig, height=360, barmode="group")


def shap_strip_figure(values: pd.DataFrame, features: pd.DataFrame, top: int = 10) -> go.Figure:
    """One row per feature, one dot per patient; colour = the patient's value of that feature."""
    order = values.abs().mean().sort_values(ascending=False).index[:top][::-1]
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for row, col in enumerate(order):
        raw = pd.to_numeric(features[col], errors="coerce")
        spread = raw.max() - raw.min()
        scaled = (raw - raw.min()) / spread if spread else raw * 0 + 0.5
        fig.add_trace(go.Scatter(
            x=values[col], y=row + rng.uniform(-0.25, 0.25, len(values)), mode="markers", showlegend=False,
            marker=dict(size=6, color=scaled, colorscale=[[0, PALETTE[0]], [1, PALETTE[1]]], opacity=0.8),
            customdata=raw, hovertemplate=f"{col} = %{{customdata}}<br>SHAP %{{x:.3f}}<extra></extra>",
        ))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=list(order), showgrid=False)
    fig.update_xaxes(title="Contribution to predicted risk (← lower risk · higher risk →)")
    return _layout(fig, height=max(260, 38 * len(order) + 90))
