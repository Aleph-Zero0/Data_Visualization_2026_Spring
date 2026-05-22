"""FreeSolv Hydration Free Energy Explorer — main app entrypoint.

Run locally with:
    python app.py

Render uses the gunicorn entrypoint declared in the Procfile (`app:server`).
"""
from __future__ import annotations

import os

import dash
import dash_bootstrap_components as dbc
from dash import dcc

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="FreeSolv Explorer",
)
server = app.server  # gunicorn entrypoint


def build_layout():
    # Sort pages by the explicit `order` set on each register_page call —
    # NOT filesystem-dependent import order (differs between macOS/Linux/Windows).
    nav_pages = sorted(
        dash.page_registry.values(), key=lambda p: p.get("order", 999)
    )
    return dbc.Container(
        [
            dbc.NavbarSimple(
                children=[
                    dbc.NavLink(p["name"], href=p["path"], active="exact")
                    for p in nav_pages
                ],
                brand="FreeSolv Hydration Free Energy Explorer",
                brand_href="/",
                color="primary",
                dark=True,
                fluid=True,
                className="mb-3",
            ),
            dcc.Store(id="selected-mol", storage_type="session"),
            dcc.Location(id="url"),
            dash.page_container,
        ],
        fluid=True,
    )


app.layout = build_layout


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
