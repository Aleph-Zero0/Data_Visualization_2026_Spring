"""SMILES → base64 PNG via RDKit Draw, with an in-memory cache so each unique
structure is rendered at most once per process.
"""
from __future__ import annotations

import base64
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import Draw

_CACHE: dict[tuple[str, int, int], str] = {}


def smiles_to_data_url(smiles: str, size: tuple[int, int] = (300, 300)) -> str:
    """Render a SMILES string to a PNG data URL suitable for html.Img(src=...)."""
    key = (smiles, size[0], size[1])
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""  # caller decides how to handle (typically: empty <img> or placeholder)
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    data_url = f"data:image/png;base64,{encoded}"
    _CACHE[key] = data_url
    return data_url
