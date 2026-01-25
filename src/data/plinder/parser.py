
"""Parsing utilities for PLINDER system structures.

We use gemmi for mmCIF parsing (fast, robust).

This module provides:
- load_structure(path) -> gemmi.Structure
- select_first_model(structure)
- structure_atoms(structure) -> list of (element, x,y,z, residue info)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import gemmi
except Exception as e:
    gemmi = None

@dataclass
class AtomRec:
    element: str
    x: float
    y: float
    z: float
    resname: str
    chain: str
    resid: int
    atom_name: str
    is_backbone: bool

BACKBONE = {"N", "CA", "C", "O", "OXT"}

def load_structure(path: str):
    if gemmi is None:
        raise ImportError("gemmi is required to parse mmCIF. Install via conda-forge (recommended).")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    st = gemmi.read_structure(str(p))
    return st

def select_first_model(st):
    # Gemmi stores models; we take model 0
    if len(st) == 0:
        raise ValueError("Empty structure")
    return st[0]

def iter_atoms(st, include_hydrogens: bool=False):
    model = select_first_model(st)
    for chain in model:
        for res in chain:
            resname = res.name
            resid = res.seqid.num
            for at in res:
                el = at.element.name.upper()
                if not include_hydrogens and el == "H":
                    continue
                pos = at.pos
                yield AtomRec(
                    element=el,
                    x=float(pos.x), y=float(pos.y), z=float(pos.z),
                    resname=resname,
                    chain=str(chain.name),
                    resid=int(resid),
                    atom_name=str(at.name).strip(),
                    is_backbone=(str(at.name).strip() in BACKBONE),
                )
