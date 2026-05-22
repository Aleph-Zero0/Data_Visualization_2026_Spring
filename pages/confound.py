"""Page 3 — Confound Revealer.

Is the apparent chemistry-vs-error correlation real, or driven by |ΔG| scale?
Toggle "Control for |ΔG|" to residualize both axes and watch r collapse.
"""
from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from data_io import DESCRIPTOR_LABELS, DESCRIPTORS, molecules

dash.register_page(
    __name__,
    name="Confound Revealer",
    path="/confound",
    order=3,
)

X_OPTIONS = [{"label": DESCRIPTOR_LABELS[d], "value": d} for d in DESCRIPTORS]


def layout():
    return dbc.Container(
        [
            html.H3("Confound Revealer"),
            html.P(
                "TPSA looks correlated with absolute prediction error (r ≈ 0.36). But "
                "molecules with bigger |ΔG| have bigger absolute errors just because the "
                "numbers are bigger — and polar molecules tend to have bigger |ΔG|. Toggle "
                "\"Control for |ΔG|\" to residualize both axes on |ΔG_expt| and see how "
                "much of the apparent chemistry effect survives.",
                className="text-muted",
            ),
            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Descriptor (x-axis)", className="fw-semibold small"),
                                    dcc.Dropdown(
                                        id="conf-x",
                                        options=X_OPTIONS,
                                        value="TPSA",
                                        clearable=False,
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Control for |ΔG_expt|", className="fw-semibold small"),
                                    dbc.Switch(
                                        id="conf-control",
                                        label="Residualize both axes on |ΔG_expt|",
                                        value=False,
                                        className="mt-2",
                                    ),
                                ],
                                md=6,
                            ),
                        ],
                        className="g-3",
                    )
                ),
                className="mb-3 shadow-sm",
            ),
            html.Div(id="conf-readout", className="text-center mb-2"),
            dcc.Graph(id="conf-scatter", config={"displayModeBar": False}),
        ],
        fluid=True,
    )


def _build_scatter(descriptor: str, control: bool) -> tuple[go.Figure, float, int]:
    df = molecules
    if control:
        x_col = f"{descriptor}_resid"
        y_col = "abs_residual_resid"
        x_label = f"{DESCRIPTOR_LABELS[descriptor]}, controlled for |ΔG_expt|"
        y_label = "Absolute prediction error, controlled for |ΔG_expt|"
    else:
        x_col = descriptor
        y_col = "abs_residual"
        x_label = DESCRIPTOR_LABELS[descriptor]
        y_label = "Absolute prediction error (kcal/mol)"

    df = df.assign(
        _x_d=df[x_col].round(3),
        _y_d=df[y_col].round(3),
    )

    fig = px.scatter(
        df,
        x=x_col, y=y_col,
        color="display_group",
        category_orders={"display_group": sorted(df["display_group"].unique())},
        labels={x_col: x_label, y_col: y_label, "display_group": "Functional group"},
        custom_data=["mol_idx", "display_name", "_x_d", "_y_d"],
        trendline="ols",
        trendline_scope="overall",
        trendline_color_override="#444",
    )
    fig.update_traces(
        marker=dict(size=6, opacity=0.7, line=dict(width=0.3, color="grey")),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            f"{x_label}: %{{customdata[2]}}<br>"
            f"{y_label}: %{{customdata[3]}}"
            "<extra></extra>"
        ),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=20, b=10),
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    r = float(np.corrcoef(df[x_col], df[y_col])[0, 1])
    return fig, r, len(df)


def _readout(r_raw: float, r_ctrl: float, n: int, control: bool) -> html.Div:
    """Show both r values side-by-side; bold + primary color on the active one."""
    raw_cls = "fw-bold text-primary" if not control else "text-muted"
    ctrl_cls = "fw-bold text-primary" if control else "text-muted"
    return html.Div(
        [
            html.Span("Pearson r — ", className="text-muted"),
            html.Span("Raw: ", className="text-muted"),
            html.Span(f"{r_raw:+.3f}", className=raw_cls),
            html.Span("   |   ", className="text-muted"),
            html.Span("Controlled: ", className="text-muted"),
            html.Span(f"{r_ctrl:+.3f}", className=ctrl_cls),
            html.Span(f"   |   n = {n}", className="text-muted ms-3"),
        ],
        className="fs-5",
    )


@callback(
    Output("conf-scatter", "figure"),
    Output("conf-readout", "children"),
    Input("conf-x", "value"),
    Input("conf-control", "value"),
)
def _update(descriptor, control):
    # Compute both r values so the readout always shows raw + controlled side-by-side.
    fig, r_active, n = _build_scatter(descriptor, control)
    if control:
        r_raw = float(np.corrcoef(molecules[descriptor], molecules["abs_residual"])[0, 1])
        r_ctrl = r_active
    else:
        r_raw = r_active
        r_ctrl = float(
            np.corrcoef(molecules[f"{descriptor}_resid"], molecules["abs_residual_resid"])[0, 1]
        )
    return fig, _readout(r_raw, r_ctrl, n, control)
