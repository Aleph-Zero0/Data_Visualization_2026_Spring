"""Page 4 — Molecule Profile.

A single molecule, profiled: 2D structure, identity, descriptors, expt vs calc,
and top-3 nearest neighbors by RDKit fingerprint (Tanimoto).
"""
from __future__ import annotations

import urllib.parse

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update

from components.kpi import kpi_card
from components.structure import smiles_to_data_url
from data_io import (
    DEFAULT_MOL_IDX,
    DESCRIPTOR_LABELS,
    DESCRIPTORS,
    get_molecule,
    get_neighbors,
    molecules,
)

dash.register_page(
    __name__,
    name="Molecule Profile",
    path="/profile",
    order=4,
)

MOL_OPTIONS = [
    {"label": row.display_name, "value": int(row.mol_idx)}
    for row in molecules.itertuples()
]


def layout(**query):
    return dbc.Container(
        [
            html.H3("Molecule Profile"),
            html.P(
                "Pick a molecule (or land here from a point you clicked on the "
                "Accuracy Explorer) to see its structure, descriptors, and the "
                "three most similar molecules in the dataset.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="profile-dropdown",
                            options=MOL_OPTIONS,
                            value=DEFAULT_MOL_IDX,
                            clearable=False,
                            searchable=True,
                            placeholder="Search a molecule by name",
                        ),
                        md=8,
                    ),
                    dbc.Col(html.Div(id="profile-warn"), md=4),
                ],
                className="g-2 mb-3",
            ),
            html.Div(id="profile-body"),
        ],
        fluid=True,
    )


def _identity_card(row) -> dbc.Card:
    pubchem_id = row.get("PubChemID")
    if pubchem_id and str(pubchem_id) != "nan":
        pubchem_link = html.A(
            f"PubChem CID {pubchem_id} ↗",
            href=f"https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem_id}",
            target="_blank",
            className="small",
        )
    else:
        pubchem_link = html.Span("PubChem ID unavailable", className="text-muted small")

    groups_str = row.get("groups_str") or ""
    group_chips = [
        dbc.Badge(g.strip(), color="secondary", className="me-1 mb-1")
        for g in groups_str.split(",")
        if g.strip()
    ]
    if not group_chips:
        group_chips = [html.Span("(no functional group tags)", className="text-muted small")]

    descriptor_rows = [
        html.Tr(
            [
                html.Td(DESCRIPTOR_LABELS[d], className="text-muted small"),
                html.Td(f"{row[d]:.2f}", className="text-end small"),
            ]
        )
        for d in DESCRIPTORS
    ]

    name = str(row["display_name"]).strip()
    iupac = str(row["iupac"]).strip()
    iupac_line = (
        html.Div(iupac, className="text-muted small mb-2")
        if iupac and iupac.lower() != name.lower()
        else None
    )

    return dbc.Card(
        dbc.CardBody(
            [
                html.H4(name, className="card-title mb-1"),
                iupac_line,
                html.Code(row["smiles"], className="small d-block mb-3"),
                html.Div(pubchem_link, className="mb-3"),
                html.Div(group_chips, className="mb-3"),
                html.Hr(),
                html.H6("Descriptors", className="text-muted text-uppercase small"),
                html.Table(descriptor_rows, className="table table-sm mb-0"),
            ]
        ),
        className="h-100 shadow-sm",
    )


def _energetics_row(row):
    expt = float(row["expt"])
    calc = float(row["calc"])
    err = float(row["residual"])
    return dbc.Row(
        [
            dbc.Col(kpi_card("Experimental ΔG (kcal/mol)", f"{expt:+.2f}"), width=True),
            dbc.Col(kpi_card("Computed ΔG (kcal/mol)", f"{calc:+.2f}"), width=True),
            dbc.Col(
                kpi_card(
                    "Signed prediction error (kcal/mol)",
                    f"{err:+.2f}",
                ),
                width=True,
            ),
        ],
        className="g-2",
    )


def _nn_card(nn_row) -> dbc.Card:
    structure_url = smiles_to_data_url(nn_row["smiles"], size=(180, 180))
    err = float(nn_row["residual"])
    return dbc.Card(
        dbc.CardBody(
            [
                html.Img(
                    src=structure_url,
                    style={"width": "100%", "objectFit": "contain"},
                )
                if structure_url
                else html.Div("(no structure)", className="text-muted small"),
                html.H6(nn_row["display_name"], className="mt-2 mb-1"),
                html.Div(
                    f"Tanimoto similarity: {nn_row['tanimoto']:.3f}",
                    className="text-muted small",
                ),
                html.Div(
                    [
                        html.Span("Its experimental ΔG: ", className="text-muted small"),
                        html.Span(f"{float(nn_row['expt']):+.2f} kcal/mol", className="small"),
                    ]
                ),
                html.Div(
                    [
                        html.Span("Signed prediction error: ", className="text-muted small"),
                        html.Span(f"{err:+.2f} kcal/mol", className="small"),
                    ]
                ),
            ]
        ),
        className="h-100 shadow-sm",
    )


def _nn_panel(mol_idx: int) -> html.Div:
    nn = get_neighbors(mol_idx)
    if len(nn) == 0:
        return html.Div("No neighbors found.", className="text-muted")

    knn_pred = float(nn["expt"].mean())
    target_expt = float(get_molecule(mol_idx)["expt"])
    knn_err = knn_pred - target_expt

    cards = [dbc.Col(_nn_card(r), md=4) for _, r in nn.iterrows()]
    return html.Div(
        [
            html.H5(
                "Three most-similar molecules (RDKit fingerprint, Tanimoto)",
                className="mb-3",
            ),
            dbc.Row(cards, className="g-3 mb-3"),
            dbc.Alert(
                [
                    html.Strong("kNN baseline (k = 3): "),
                    html.Br(),
                    html.Span("Prediction (mean of neighbors' experimental ΔG): "),
                    html.Strong(f"{knn_pred:+.2f} kcal/mol"),
                    html.Br(),
                    html.Span("Actual experimental ΔG: "),
                    html.Strong(f"{target_expt:+.2f} kcal/mol"),
                    html.Br(),
                    html.Span("Signed prediction error: "),
                    html.Strong(
                        f"{knn_err:+.2f} kcal/mol",
                        className="text-danger" if abs(knn_err) > 2 else "",
                    ),
                ],
                color="light",
                className="small mb-0",
            ),
        ]
    )


def _profile_body(mol_idx: int) -> html.Div:
    row = get_molecule(mol_idx).to_dict()
    structure_url = smiles_to_data_url(row["smiles"], size=(320, 320))

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                html.Img(
                                    src=structure_url,
                                    style={"width": "100%", "objectFit": "contain"},
                                )
                                if structure_url
                                else html.Div("(structure unavailable)", className="text-muted"),
                            ),
                            className="h-100 shadow-sm",
                        ),
                        md=5,
                    ),
                    dbc.Col(_identity_card(row), md=7),
                ],
                className="g-3 mb-3",
            ),
            html.Div(_energetics_row(row), className="mb-3"),
            dbc.Card(
                dbc.CardBody(_nn_panel(mol_idx)),
                className="mb-3 shadow-sm",
            ),
        ]
    )


def _parse_mol_query(search: str | None) -> tuple[int, str | None]:
    """Parse `?mol=N` from the URL search string. Returns (mol_idx, warning).
    Falls back to DEFAULT_MOL_IDX on missing or invalid input.
    """
    if not search:
        return DEFAULT_MOL_IDX, None
    qs = urllib.parse.parse_qs(search.lstrip("?"))
    raw = qs.get("mol", [None])[0]
    if raw is None:
        return DEFAULT_MOL_IDX, None
    try:
        mol_idx = int(raw)
    except ValueError:
        return DEFAULT_MOL_IDX, f"Couldn't parse mol={raw!r} — showing default."
    if mol_idx not in set(molecules["mol_idx"]):
        return DEFAULT_MOL_IDX, f"mol_idx {mol_idx} not in dataset — showing default."
    return mol_idx, None


@callback(
    Output("profile-dropdown", "value"),
    Output("profile-warn", "children"),
    Input("url", "search"),
    State("selected-mol", "data"),
    prevent_initial_call=False,
)
def _sync_from_url(search, stored):
    mol_idx, warning = _parse_mol_query(search)
    # URL takes precedence; if URL had no `mol`, try the Store.
    if not search and stored is not None:
        try:
            stored_int = int(stored)
            if stored_int in set(molecules["mol_idx"]):
                mol_idx = stored_int
        except (TypeError, ValueError):
            pass
    if warning:
        warn = dbc.Alert(warning, color="warning", className="small mb-0 py-2 px-3")
    else:
        warn = None
    return mol_idx, warn


@callback(
    Output("profile-body", "children"),
    Input("profile-dropdown", "value"),
)
def _render_body(mol_idx):
    if mol_idx is None:
        return no_update
    return _profile_body(int(mol_idx))
