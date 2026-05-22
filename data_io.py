"""Shared data loader. Loads the parquet files produced by data_prep.py once
at import time and exposes the two dataframes plus convenience constants.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"

MOLECULES_PATH = PROCESSED / "molecules.parquet"
NEIGHBORS_PATH = PROCESSED / "neighbors.parquet"

if not MOLECULES_PATH.exists() or not NEIGHBORS_PATH.exists():
    raise FileNotFoundError(
        f"Processed data not found at {PROCESSED}. "
        f"Run `python data_prep.py` first."
    )

molecules: pd.DataFrame = pd.read_parquet(MOLECULES_PATH)
neighbors: pd.DataFrame = pd.read_parquet(NEIGHBORS_PATH)

# Collapse rare primary_group values (n < 10) into "other" for the UI.
# The canonical chemistry label stays in `primary_group`; `display_group` is
# only used for the dropdown options and the legend.
_GROUP_MIN = 10
_counts = molecules["primary_group"].value_counts()
_MAIN_GROUPS = set(_counts[_counts >= _GROUP_MIN].index)
molecules["display_group"] = molecules["primary_group"].where(
    molecules["primary_group"].isin(_MAIN_GROUPS), "other"
)

DESCRIPTORS = [
    "TPSA", "LogP", "HBA", "HBD", "MolWt",
    "RotBonds", "RingCount", "AromaticRings",
    "HeavyAtoms", "FractionCSP3", "NumHeteroatoms",
]

DESCRIPTOR_LABELS = {
    "TPSA": "Topological polar surface area (Å²)",
    "LogP": "Partition coefficient (LogP)",
    "HBA": "H-bond acceptors",
    "HBD": "H-bond donors",
    "MolWt": "Molecular weight (Da)",
    "RotBonds": "Rotatable bonds",
    "RingCount": "Ring count",
    "AromaticRings": "Aromatic rings",
    "HeavyAtoms": "Heavy atoms",
    "FractionCSP3": "Fraction sp³ carbons",
    "NumHeteroatoms": "Heteroatoms",
}

TARGET_LABELS = {
    "expt": "Experimental ΔG (kcal/mol)",
    "calc": "Computed ΔG (kcal/mol)",
    "residual": "Signed prediction error (kcal/mol)",
    "abs_residual": "Absolute prediction error (kcal/mol)",
}

DEFAULT_MOL_IDX = int(
    molecules.loc[molecules["display_name"].str.contains("glucose", case=False, na=False), "mol_idx"].iloc[0]
)


def get_molecule(mol_idx: int) -> pd.Series:
    """Return one row by mol_idx, or raise KeyError if not found."""
    matches = molecules[molecules["mol_idx"] == mol_idx]
    if len(matches) == 0:
        raise KeyError(f"mol_idx {mol_idx} not in molecules table")
    return matches.iloc[0]


def get_neighbors(mol_idx: int) -> pd.DataFrame:
    """Return the top-3 NN rows for a molecule, with the neighbor's data joined in."""
    rows = neighbors[neighbors["mol_idx"] == mol_idx].sort_values("nn_rank")
    return rows.merge(
        molecules[["mol_idx", "display_name", "smiles", "expt", "calc", "residual", "abs_residual"]],
        left_on="nn_idx", right_on="mol_idx", suffixes=("", "_nn"),
    )
