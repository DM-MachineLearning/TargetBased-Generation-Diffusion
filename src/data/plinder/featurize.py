
"""Featurization for pocket and ligand graphs.

Pocket atoms: element + residue info + backbone flag + donor/acceptor heuristics (simple).
Ligands: for research-grade ligand bonding and aromaticity, RDKit is required.

We keep this module lightweight but structured so you can upgrade features later.
"""
from __future__ import annotations
import numpy as np

ELEMENTS = ["C","N","O","S","P","F","CL","BR","I","FE","ZN","MG","NA","K","CA","MN","CU","CO","NI","SE","B","SI","OTHER"]
E2I = {e:i for i,e in enumerate(ELEMENTS)}

RESIDUES = [
 "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
 "SEC","PYL","UNK"
]
R2I = {r:i for i,r in enumerate(RESIDUES)}

def one_hot(i, n):
    v = np.zeros(n, dtype=np.float32)
    v[i] = 1.0
    return v

def pocket_atom_features(atom):
    el = atom.element.upper()
    if el not in E2I: el = "OTHER"
    rn = atom.resname.upper()
    if rn not in R2I: rn = "UNK"
    # Very simple donor/acceptor heuristics by residue + atom name could be added later.
    feat = np.concatenate([
        one_hot(E2I[el], len(ELEMENTS)),
        one_hot(R2I[rn], len(RESIDUES)),
        np.array([1.0 if atom.is_backbone else 0.0], dtype=np.float32),
    ], axis=0)
    return feat

def ligand_from_rdkit(mol):
    """Return (coords, atom_type_ids, edge_index, bond_type_ids)."""
    from rdkit import Chem
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    coords = np.zeros((n,3), dtype=np.float32)
    atom_types = np.zeros((n,), dtype=np.int64)
    # Map RDKit atomic number to a compact set; for now we map by element list
    for i, a in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]
        el = a.GetSymbol().upper()
        if el not in E2I: el = "OTHER"
        atom_types[i] = E2I[el] if E2I[el] < 64 else 0  # keep <=64 for toy; adjust as needed

    edge_u=[]; edge_v=[]; bond_t=[]
    for b in mol.GetBonds():
        u=b.GetBeginAtomIdx(); v=b.GetEndAtomIdx()
        # undirected -> add both
        edge_u += [u,v]; edge_v += [v,u]
        bt = int(b.GetBondTypeAsDouble())  # 1,2,3,1.5
        # discretize
        if bt == 1: bt_id = 0
        elif bt == 2: bt_id = 1
        elif bt == 3: bt_id = 2
        else: bt_id = 3  # aromatic/other
        bond_t += [bt_id, bt_id]
    edge_index = np.stack([np.array(edge_u, dtype=np.int64), np.array(edge_v, dtype=np.int64)], axis=0)
    bond_type = np.array(bond_t, dtype=np.int64)
    return coords, atom_types, edge_index, bond_type
