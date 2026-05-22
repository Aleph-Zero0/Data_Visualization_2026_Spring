"""Page 1 — Accuracy Explorer (landing page).

How close is the simulation to experiment?
Filterable Calc-vs-Expt scatter with live KPI cards.
"""
from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update

from components.kpi import kpi_card
from components.structure import smiles_to_data_url
from data_io import DESCRIPTOR_LABELS, molecules

dash.register_page(
    __name__,
    name="Accuracy Explorer",
    path="/",
    order=1,
)

import math

GROUPS = sorted(
    [g for g in molecules["display_group"].unique() if g != "other"]
) + ["other"]


def _round_down(x: float, step: float) -> float:
    return math.floor(x / step) * step


def _round_up(x: float, step: float) -> float:
    return math.ceil(x / step) * step


# Rounded slider bounds so the tooltips and end marks show clean numbers
# instead of raw 4-decimal extremes from the data.
TPSA_MIN = _round_down(float(molecules["TPSA"].min()), 10)
TPSA_MAX = _round_up(float(molecules["TPSA"].max()), 10)
LOGP_MIN = _round_down(float(molecules["LogP"].min()), 1)
LOGP_MAX = _round_up(float(molecules["LogP"].max()), 1)
MW_MIN = _round_down(float(molecules["MolWt"].min()), 50)
MW_MAX = _round_up(float(molecules["MolWt"].max()), 50)
ERR_MAX = _round_up(float(molecules["abs_residual"].max()), 1)


def _int_marks(lo: float, hi: float, step: int) -> dict:
    """Marks at integer multiples of `step` across [lo, hi]."""
    start = int(math.ceil(lo / step) * step)
    end = int(math.floor(hi / step) * step)
    return {v: str(v) for v in range(start, end + 1, step)}


def _kpi_row():
    return dbc.Row(
        [
            dbc.Col(kpi_card("Molecules", "—"), id="kpi-n", width=True),
            dbc.Col(kpi_card("MAE (kcal/mol)", "—"), id="kpi-mae", width=True),
            dbc.Col(kpi_card("RMSE (kcal/mol)", "—"), id="kpi-rmse", width=True),
            dbc.Col(kpi_card("R² (calc vs expt)", "—"), id="kpi-r2", width=True),
            dbc.Col(kpi_card("Bias (mean signed prediction error)", "—"), id="kpi-bias", width=True),
        ],
        className="g-2 mb-2",
        id="kpi-row",
    )


def _filter_card():
    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Label("Functional group", className="fw-semibold small me-2"),
                                    dbc.Button(
                                        "All", id="acc-group-all", color="link",
                                        size="sm", className="p-0 small me-2",
                                    ),
                                    dbc.Button(
                                        "Clear", id="acc-group-clear", color="link",
                                        size="sm", className="p-0 small",
                                    ),
                                ],
                                className="d-flex align-items-center",
                            ),
                            dcc.Dropdown(
                                id="acc-group",
                                options=[{"label": g, "value": g} for g in GROUPS],
                                value=[],
                                multi=True,
                                placeholder="All groups",
                            ),
                            html.Label("Color points by", className="fw-semibold small mt-2"),
                            dbc.RadioItems(
                                id="acc-color",
                                options=[
                                    {"label": "Functional group", "value": "primary_group"},
                                    {"label": "Absolute prediction error", "value": "abs_residual"},
                                ],
                                value="primary_group",
                                inline=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Max absolute prediction error (kcal/mol)", className="fw-semibold small"),
                            dcc.Slider(
                                id="acc-err",
                                min=0, max=ERR_MAX, step=0.1, value=ERR_MAX,
                                marks=_int_marks(0, ERR_MAX, 2),
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Label("TPSA range (Å²)", className="fw-semibold small mt-1"),
                            dcc.RangeSlider(
                                id="acc-tpsa",
                                min=TPSA_MIN, max=TPSA_MAX, step=1,
                                value=[TPSA_MIN, TPSA_MAX],
                                marks=_int_marks(TPSA_MIN, TPSA_MAX, 50),
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("LogP range", className="fw-semibold small"),
                            dcc.RangeSlider(
                                id="acc-logp",
                                min=LOGP_MIN, max=LOGP_MAX, step=0.1,
                                value=[LOGP_MIN, LOGP_MAX],
                                marks=_int_marks(LOGP_MIN, LOGP_MAX, 2),
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Label("Molecular weight (Da)", className="fw-semibold small mt-1"),
                            dcc.RangeSlider(
                                id="acc-mw",
                                min=MW_MIN, max=MW_MAX, step=1,
                                value=[MW_MIN, MW_MAX],
                                marks=_int_marks(MW_MIN, MW_MAX, 100),
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="g-3",
            )
        ),
        className="mb-2 shadow-sm",
    )


def _empty_side_panel():
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Molecule inspector", className="card-title"),
                html.P(
                    "Click a point on the scatter to inspect a molecule.",
                    className="text-muted",
                ),
            ]
        ),
        className="h-100 shadow-sm",
        id="side-panel-card",
    )


def layout():
    return dbc.Container(
        [
            html.Div(
                [
                    html.H4("Accuracy Explorer", className="mb-0 d-inline-block me-3"),
                    html.Span(
                        "How close is the simulation to experiment?",
                        className="text-muted small me-3",
                    ),
                    dbc.Button(
                        "Reset filters",
                        id="acc-reset",
                        color="outline-secondary",
                        size="sm",
                    ),
                ],
                className="mb-2 d-flex align-items-center",
            ),
            _kpi_row(),
            _filter_card(),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id="acc-scatter", config={"displayModeBar": False}),
                        lg=8,
                    ),
                    dbc.Col(html.Div(_empty_side_panel(), id="side-panel"), lg=4),
                ],
                className="g-3",
            ),
        ],
        fluid=True,
    )


def _filter_df(group, err_max, tpsa_range, logp_range, mw_range) -> pd.DataFrame:
    df = molecules
    if group:
        df = df[df["display_group"].isin(group)]
    df = df[df["abs_residual"] <= err_max]
    df = df[df["TPSA"].between(*tpsa_range)]
    df = df[df["LogP"].between(*logp_range)]
    df = df[df["MolWt"].between(*mw_range)]
    return df


def _build_scatter(df: pd.DataFrame, color_by: str) -> go.Figure:
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No molecules match these filters.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="grey"),
        )
        fig.update_layout(height=520, template="plotly_white")
        return fig

    # Pre-round the values used in hovertemplate. Plotly's hovertemplate
    # format-spec on customdata is unreliable across versions, so we round
    # the source data instead of relying on display-time formatting.
    df = df.assign(
        _residual_d=df["residual"].round(2),
        _abs_residual_d=df["abs_residual"].round(2),
    )

    common = dict(
        custom_data=["mol_idx", "display_name", "primary_group", "_residual_d", "_abs_residual_d"],
    )
    if color_by == "abs_residual":
        fig = px.scatter(
            df, x="expt", y="calc",
            color="abs_residual",
            color_continuous_scale="Viridis",
            range_color=[0, ERR_MAX],
            labels={
                "expt": "Experimental ΔG (kcal/mol)",
                "calc": "Computed ΔG (kcal/mol)",
                "abs_residual": "Abs. prediction error",
            },
            **common,
        )
    else:
        fig = px.scatter(
            df, x="expt", y="calc",
            color="display_group",
            labels={
                "expt": "Experimental ΔG (kcal/mol)",
                "calc": "Computed ΔG (kcal/mol)",
                "display_group": "Functional group",
            },
            category_orders={"display_group": GROUPS},
            **common,
        )

    fig.update_traces(
        marker=dict(size=7, opacity=0.75, line=dict(width=0.4, color="grey")),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Experimental ΔG: %{x:.2f} kcal/mol<br>"
            "Computed ΔG: %{y:.2f} kcal/mol<br>"
            "Signed prediction error: %{customdata[3]} kcal/mol<br>"
            "Functional group: %{customdata[2]}"
            "<extra></extra>"
        ),
    )

    # y=x diagonal
    lo = float(min(df["expt"].min(), df["calc"].min()))
    hi = float(max(df["expt"].max(), df["calc"].max()))
    pad = 0.05 * (hi - lo)
    fig.add_trace(
        go.Scatter(
            x=[lo - pad, hi + pad], y=[lo - pad, hi + pad],
            mode="lines",
            line=dict(color="lightgrey", dash="dash", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    return fig


def _metrics(df: pd.DataFrame) -> dict[str, str]:
    if len(df) == 0:
        return dict(n="0", mae="—", rmse="—", r2="—", bias="—")
    err = df["residual"].to_numpy()
    n = len(df)
    mae = np.mean(np.abs(err))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    if n >= 2 and df["expt"].std() > 0 and df["calc"].std() > 0:
        r = float(df["expt"].corr(df["calc"]))
        r2 = f"{r**2:.3f}"
    else:
        r2 = "—"
    return dict(
        n=f"{n}",
        mae=f"{mae:.3f}",
        rmse=f"{rmse:.3f}",
        r2=r2,
        bias=f"{bias:+.3f}",
    )


@callback(
    Output("acc-scatter", "figure"),
    Output("kpi-n", "children"),
    Output("kpi-mae", "children"),
    Output("kpi-rmse", "children"),
    Output("kpi-r2", "children"),
    Output("kpi-bias", "children"),
    Input("acc-group", "value"),
    Input("acc-err", "value"),
    Input("acc-tpsa", "value"),
    Input("acc-logp", "value"),
    Input("acc-mw", "value"),
    Input("acc-color", "value"),
)
def _update_scatter(group, err_max, tpsa_range, logp_range, mw_range, color_by):
    df = _filter_df(group, err_max, tpsa_range, logp_range, mw_range)
    fig = _build_scatter(df, color_by)
    m = _metrics(df)
    return (
        fig,
        kpi_card("Molecules", m["n"]),
        kpi_card("MAE (kcal/mol)", m["mae"]),
        kpi_card("RMSE (kcal/mol)", m["rmse"]),
        kpi_card("R² (calc vs expt)", m["r2"]),
        kpi_card("Bias (mean signed prediction error)", m["bias"]),
    )


def _side_panel_for(mol_idx: int):
    row = molecules[molecules["mol_idx"] == mol_idx].iloc[0]

    descriptor_rows = [
        html.Tr([html.Td(DESCRIPTOR_LABELS[d], className="text-muted small"),
                 html.Td(f"{row[d]:.2f}", className="text-end small")])
        for d in ["TPSA", "LogP", "HBA", "HBD", "MolWt"]
    ]

    structure_url = smiles_to_data_url(row["smiles"], size=(220, 180))
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(row["display_name"], className="card-title mb-1"),
                html.Div(row["primary_group"], className="text-muted small mb-2"),
                html.Div(
                    html.Img(
                        src=structure_url,
                        style={"maxWidth": "100%", "objectFit": "contain"},
                    ),
                    className="text-center mb-2",
                ) if structure_url else None,
                html.Code(row["smiles"], className="small d-block mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div("Experimental ΔG", className="text-muted small"),
                                html.Div(f"{row['expt']:+.2f}", className="fw-bold"),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.Div("Computed ΔG", className="text-muted small"),
                                html.Div(f"{row['calc']:+.2f}", className="fw-bold"),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.Div("Signed prediction error", className="text-muted small"),
                                html.Div(
                                    f"{row['residual']:+.2f}",
                                    className="fw-bold "
                                    + ("text-danger" if abs(row["residual"]) > 2 else ""),
                                ),
                            ]
                        ),
                    ],
                    className="g-2 mb-3",
                ),
                html.Table(
                    descriptor_rows,
                    className="table table-sm mb-3",
                ),
                dbc.Button(
                    "View full profile →",
                    id="view-profile-btn",
                    color="primary",
                    size="sm",
                    className="w-100",
                ),
            ]
        ),
        className="h-100 shadow-sm",
    )


@callback(
    Output("side-panel", "children"),
    Output("selected-mol", "data"),
    Input("acc-scatter", "clickData"),
    prevent_initial_call=True,
)
def _on_point_click(click_data):
    if not click_data or not click_data.get("points"):
        return no_update, no_update
    point = click_data["points"][0]
    customdata = point.get("customdata")
    if not customdata:
        return no_update, no_update
    mol_idx = int(customdata[0])
    return _side_panel_for(mol_idx), mol_idx


@callback(
    Output("url", "href", allow_duplicate=True),
    Input("view-profile-btn", "n_clicks"),
    State("selected-mol", "data"),
    prevent_initial_call=True,
)
def _on_view_profile(n_clicks, mol_idx):
    if not n_clicks or mol_idx is None:
        return no_update
    return f"/profile?mol={int(mol_idx)}"


@callback(
    Output("acc-group", "value"),
    Input("acc-group-all", "n_clicks"),
    Input("acc-group-clear", "n_clicks"),
    prevent_initial_call=True,
)
def _on_group_button(n_all, n_clear):
    from dash import ctx
    if ctx.triggered_id == "acc-group-all":
        return list(GROUPS)
    if ctx.triggered_id == "acc-group-clear":
        return []
    return no_update


@callback(
    Output("acc-group", "value", allow_duplicate=True),
    Output("acc-err", "value"),
    Output("acc-tpsa", "value"),
    Output("acc-logp", "value"),
    Output("acc-mw", "value"),
    Output("acc-color", "value"),
    Input("acc-reset", "n_clicks"),
    prevent_initial_call=True,
)
def _on_reset(n_clicks):
    if not n_clicks:
        return (no_update,) * 6
    return (
        [],                              # group (empty = all)
        ERR_MAX,                         # err threshold
        [TPSA_MIN, TPSA_MAX],            # TPSA range
        [LOGP_MIN, LOGP_MAX],            # LogP range
        [MW_MIN, MW_MAX],                # MolWt range
        "primary_group",                 # color radio
    )
