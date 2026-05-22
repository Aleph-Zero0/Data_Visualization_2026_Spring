"""Page 2 — Descriptor Sandbox.

What molecular properties predict hydration free energy?
Pick any X / Y / color from the dropdowns; click a heatmap cell to load a pair.
"""
from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html, no_update

from data_io import DESCRIPTOR_LABELS, DESCRIPTORS, TARGET_LABELS, molecules

dash.register_page(
    __name__,
    name="Descriptor Sandbox",
    path="/descriptors",
    order=2,
)

X_OPTIONS = [{"label": DESCRIPTOR_LABELS[d], "value": d} for d in DESCRIPTORS]
Y_OPTIONS = [
    {"label": TARGET_LABELS[t], "value": t}
    for t in ["expt", "calc", "residual", "abs_residual"]
]
COLOR_OPTIONS = [
    {"label": "(none)", "value": "none"},
    {"label": "Functional group", "value": "display_group"},
    {"label": "Absolute prediction error", "value": "abs_residual"},
]


def _all_labels() -> dict[str, str]:
    return {**DESCRIPTOR_LABELS, **TARGET_LABELS}


def layout():
    return dbc.Container(
        [
            html.H3("Descriptor Sandbox"),
            html.P(
                "What molecular properties predict hydration free energy? "
                "Pick any two columns, or click a heatmap cell on the right to load a pair.",
                className="text-muted",
            ),
            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("X-axis", className="fw-semibold small"),
                                    dcc.Dropdown(
                                        id="desc-x",
                                        options=X_OPTIONS,
                                        value="TPSA",
                                        clearable=False,
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Y-axis", className="fw-semibold small"),
                                    dcc.Dropdown(
                                        id="desc-y",
                                        options=Y_OPTIONS,
                                        value="expt",
                                        clearable=False,
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Color by", className="fw-semibold small"),
                                    dcc.Dropdown(
                                        id="desc-color",
                                        options=COLOR_OPTIONS,
                                        value="display_group",
                                        clearable=False,
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Trendline", className="fw-semibold small"),
                                    dbc.RadioItems(
                                        id="desc-trend",
                                        options=[
                                            {"label": "None", "value": "none"},
                                            {"label": "OLS", "value": "ols"},
                                            {"label": "LOWESS", "value": "lowess"},
                                        ],
                                        value="none",
                                        inline=True,
                                    ),
                                ],
                                md=3,
                            ),
                        ],
                        className="g-3",
                    )
                ),
                className="mb-3 shadow-sm",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(id="desc-readout", className="text-center mb-2"),
                            dcc.Graph(id="desc-scatter", config={"displayModeBar": False}),
                        ],
                        lg=7,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                "Correlation heatmap — click a cell to load that pair into the scatter.",
                                className="text-muted small mb-2",
                            ),
                            dcc.Graph(id="desc-heatmap", config={"displayModeBar": False}),
                        ],
                        lg=5,
                    ),
                ],
                className="g-3",
            ),
        ],
        fluid=True,
    )


def _readout(r: float | None, n: int) -> html.Div:
    if r is None:
        return html.Div(f"n = {n}", className="text-muted small")
    return html.Div(
        [
            html.Span("Pearson r = ", className="text-muted"),
            html.Span(f"{r:+.3f}", className="fw-bold"),
            html.Span(f"   |   n = {n}", className="text-muted"),
        ],
        className="fs-5",
    )


def _scatter(x: str, y: str, color: str, trend: str) -> tuple[go.Figure, float | None, int]:
    labels = _all_labels()
    df = molecules.copy()
    df = df.assign(
        _x_d=df[x].round(3),
        _y_d=df[y].round(3),
    )
    n = len(df)

    px_kwargs = dict(
        x=x, y=y,
        labels={x: labels.get(x, x), y: labels.get(y, y)},
        custom_data=["mol_idx", "display_name", "_x_d", "_y_d"],
        trendline=trend if trend in ("ols", "lowess") else None,
        trendline_scope="overall",  # one line across all data, not per color group
        trendline_color_override="#444",
    )

    if color == "none":
        fig = px.scatter(df, **px_kwargs)
        fig.update_traces(marker=dict(color="#3b6ea0"))
    elif color == "display_group":
        fig = px.scatter(df, color="display_group", **px_kwargs,
                         category_orders={"display_group": sorted(df["display_group"].unique())})
    else:
        fig = px.scatter(df, color=color, color_continuous_scale="Viridis", **px_kwargs)

    # Hovertemplate that uses the rounded customdata (avoid Plotly format-spec quirks).
    fig.update_traces(
        marker=dict(size=6, opacity=0.7, line=dict(width=0.3, color="grey")),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            f"{labels.get(x, x)}: %{{customdata[2]}}<br>"
            f"{labels.get(y, y)}: %{{customdata[3]}}"
            "<extra></extra>"
        ),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white",
    )

    # Pearson r on the pair
    r = float(np.corrcoef(df[x], df[y])[0, 1]) if df[x].std() > 0 and df[y].std() > 0 else None
    return fig, r, n


TARGETS = ["expt", "calc", "residual", "abs_residual"]
TARGET_SHORT = {
    "expt": "ΔG_expt",
    "calc": "ΔG_calc",
    "residual": "signed err",
    "abs_residual": "|err|",
}
TARGET_SHORT_INV = {v: k for k, v in TARGET_SHORT.items()}


def _heatmap() -> go.Figure:
    mat = np.zeros((len(DESCRIPTORS), len(TARGETS)))
    for i, d in enumerate(DESCRIPTORS):
        for j, t in enumerate(TARGETS):
            mat[i, j] = molecules[d].corr(molecules[t])

    fig = go.Figure(
        go.Heatmap(
            z=mat,
            x=[TARGET_SHORT[t] for t in TARGETS],
            y=DESCRIPTORS,
            colorscale="RdBu",
            zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:+.2f}" for v in row] for row in mat],
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar=dict(title="r", thickness=12),
            hovertemplate=(
                "<b>%{y}</b> × <b>%{x}</b><br>r = %{z:+.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white",
        xaxis=dict(side="top", fixedrange=True),
        yaxis=dict(autorange="reversed", fixedrange=True),
        dragmode=False,
    )
    return fig


@callback(
    Output("desc-scatter", "figure"),
    Output("desc-readout", "children"),
    Input("desc-x", "value"),
    Input("desc-y", "value"),
    Input("desc-color", "value"),
    Input("desc-trend", "value"),
)
def _update_scatter(x, y, color, trend):
    fig, r, n = _scatter(x, y, color, trend)
    return fig, _readout(r, n)


@callback(
    Output("desc-heatmap", "figure"),
    Input("desc-x", "value"),  # cheap input just to ensure heatmap renders on load
)
def _update_heatmap(_):
    return _heatmap()


@callback(
    Output("desc-x", "value"),
    Output("desc-y", "value"),
    Input("desc-heatmap", "clickData"),
    prevent_initial_call=True,
)
def _on_heatmap_click(click_data):
    if not click_data or not click_data.get("points"):
        return no_update, no_update
    point = click_data["points"][0]
    descriptor = point.get("y")
    target_short = point.get("x")
    target = TARGET_SHORT_INV.get(target_short)
    if descriptor not in DESCRIPTORS or target is None:
        return no_update, no_update
    return descriptor, target
