"""KPI card factory — used for the N / MAE / RMSE / R² / Bias row on Page 1."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def kpi_card(label: str, value: str, *, helptext: str | None = None) -> dbc.Card:
    body = [
        html.Div(label, className="text-muted text-uppercase small fw-semibold"),
        html.Div(value, className="fs-4 fw-bold"),
    ]
    if helptext:
        body.append(html.Div(helptext, className="text-muted small"))
    return dbc.Card(dbc.CardBody(body), className="h-100 shadow-sm")
