
"""PLINDER system -> graph tensors for TaPR-Diff.

This module:
- locates receptor/system files inside a system directory
- parses protein atoms (gemmi)
- parses ligand with RDKit when available (preferred)
- extracts pocket atoms within cutoff of ligand coords
- builds pocket KNN edges, ligand bond edges, and ligand-pocket cross KNN edges
- saves *.pt artifacts (torch)

Design principle: fail loudly with actionable errors (missing files, RDKit absent, etc.).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import torch
import glob

from .parser import load_structure, iter_atoms
from .featurize import pocket_atom_features, ligand_from_rdkit

def _find_first(system_dir: Path, patterns: List[str]) -> Path:
    for pat in patterns:
        hits = list(system_dir.glob(pat))
        if hits:
            return hits[0]
    # also search recursively for robustness
    for pat in patterns:
        hits = glob.glob(str(system_dir / "**" / pat), recursive=True)
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"Could not find any of {patterns} under {system_dir}")

def _load_ligand(system_dir: Path):
    # Try common ligand formats
    # PLINDER systems often include system.cif containing ligand; but bond orders are best from SDF if present.
    # Also check ligand_files/ subdirectory (common in PLINDER)
    sdf_patterns = [
        "*ligand*.sdf", "*ligands*.sdf", "*ligand*.mol", "*ligand*.mol2",
        "ligand.sdf", "ligand.mol", "ligand.mol2",
        "ligand_files/*.sdf", "ligand_files/*.mol", "ligand_files/*.mol2",
    ]
    try:
        lig_path = _find_first(system_dir, sdf_patterns)
    except Exception:
        lig_path = None

    try:
        from rdkit import Chem
    except Exception as e:
        raise ImportError("RDKit is required for ligand parsing with bonds. Install via conda-forge rdkit.") from e

    if lig_path is not None:
        ext = lig_path.suffix.lower()
        if ext == ".sdf":
            suppl = Chem.SDMolSupplier(str(lig_path), removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        elif ext == ".mol2":
            mol = Chem.MolFromMol2File(str(lig_path), removeHs=False)
        else:
            mol = Chem.MolFromMolFile(str(lig_path), removeHs=False)
        if mol is None:
            raise ValueError(f"Failed to parse ligand file {lig_path}")
        if mol.GetNumConformers() == 0:
            raise ValueError(f"Ligand has no 3D conformer in {lig_path}")
        return mol, lig_path

    # Fallback: attempt to extract ligand from system.cif (bond orders may be poor). Not implemented fully.
    raise FileNotFoundError("No ligand SDF/MOL/MOL2 found. Provide ligand file or extend extractor from system.cif.")

def knn_edges(X: np.ndarray, k: int) -> np.ndarray:
    # X: [N,3] -> edge_index [2, N*k] directed
    N = X.shape[0]
    # compute distances
    d2 = ((X[:,None,:] - X[None,:,:])**2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    src = np.repeat(np.arange(N), k)
    dst = nn.reshape(-1)
    return np.stack([src, dst], axis=0).astype(np.int64)

def cross_knn_edges(Xl: np.ndarray, Xp: np.ndarray, k: int) -> np.ndarray:
    # ligand->pocket edges: [2, Nl*k] where src in ligand, dst in pocket
    Nl = Xl.shape[0]
    d2 = ((Xl[:,None,:] - Xp[None,:,:])**2).sum(-1)  # [Nl,Np]
    nn = np.argsort(d2, axis=1)[:, :k]
    src = np.repeat(np.arange(Nl), k)
    dst = nn.reshape(-1)
    return np.stack([src, dst], axis=0).astype(np.int64)

def system_to_graph(
    system_dir: Path,
    out_dir: Path,
    pocket_cutoff_A: float = 10.0,
    pocket_knn: int = 24,
    cross_knn: int = 32,
    include_hydrogens: bool = False,
    max_pocket_atoms: int = 2048,
    max_ligand_atoms: int = 256,
) -> Dict:
    system_dir = Path(system_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find receptor file (prefer receptor.cif; fallback to system.cif)
    receptor_path = None
    for pat in ["receptor.cif", "*receptor*.cif", "protein.cif", "*protein*.cif"]:
        hits = list(system_dir.glob(pat))
        if hits:
            receptor_path = hits[0]; break
    if receptor_path is None:
        # recursive search
        for pat in ["receptor.cif", "*receptor*.cif", "protein.cif", "*protein*.cif"]:
            hits = glob.glob(str(system_dir / "**" / pat), recursive=True)
            if hits:
                receptor_path = Path(hits[0]); break
    if receptor_path is None:
        raise FileNotFoundError(f"No receptor/protein cif found under {system_dir}")

    st = load_structure(str(receptor_path))
    atoms = list(iter_atoms(st, include_hydrogens=include_hydrogens))
    Xprot = np.array([[a.x,a.y,a.z] for a in atoms], dtype=np.float32)
    Hprot = np.stack([pocket_atom_features(a) for a in atoms], axis=0).astype(np.float32)

    lig_mol, lig_path = _load_ligand(system_dir)
    Xlig, A, edge_l, bond_t = ligand_from_rdkit(lig_mol)

    if Xlig.shape[0] > max_ligand_atoms:
        raise ValueError(f"Ligand too large ({Xlig.shape[0]} atoms) > max_ligand_atoms={max_ligand_atoms}")

    # Extract pocket atoms within cutoff of ligand atoms
    # Compute min distance from each protein atom to ligand
    d2 = ((Xprot[:,None,:] - Xlig[None,:,:])**2).sum(-1)  # [Np, Nl]
    mind = np.sqrt(d2.min(axis=1))
    pocket_idx = np.where(mind <= pocket_cutoff_A)[0]
    if pocket_idx.size == 0:
        raise ValueError("No pocket atoms within cutoff; check units/cutoff.")

    # Optionally cap pocket size by taking closest atoms
    if pocket_idx.size > max_pocket_atoms:
        order = np.argsort(mind[pocket_idx])[:max_pocket_atoms]
        pocket_idx = pocket_idx[order]

    Xp = Xprot[pocket_idx]
    Hp = Hprot[pocket_idx]

    edge_p = knn_edges(Xp, min(pocket_knn, max(1, Xp.shape[0]-1)))
    edge_lp = cross_knn_edges(Xlig, Xp, min(cross_knn, Xp.shape[0]))

    # Edge attributes (distances and relative vectors)
    def edge_attr(Xsrc, Xdst, edge_index):
        src = edge_index[0]; dst = edge_index[1]
        rel = Xdst[dst] - Xsrc[src]
        dist = np.linalg.norm(rel, axis=1, keepdims=True)
        return np.concatenate([dist, rel], axis=1).astype(np.float32)

    ep_attr = edge_attr(Xp, Xp, edge_p)
    elp_attr = edge_attr(Xlig, Xp, edge_lp)

    # Save artifacts
    torch.save(dict(X=torch.tensor(Xlig), A=torch.tensor(A), edge_index=torch.tensor(edge_l), bond_type=torch.tensor(bond_t)),
               out_dir / "ligand_graph.pt")
    torch.save(dict(Xp=torch.tensor(Xp), Hp=torch.tensor(Hp), edge_index=torch.tensor(edge_p), edge_attr=torch.tensor(ep_attr)),
               out_dir / "pocket_graph.pt")
    torch.save(dict(edge_index=torch.tensor(edge_lp), edge_attr=torch.tensor(elp_attr), pocket_index=torch.tensor(pocket_idx)),
               out_dir / "cross_edges.pt")

    return dict(
        system_dir=str(system_dir),
        receptor_path=str(receptor_path),
        ligand_path=str(lig_path),
        n_protein_atoms=int(Xprot.shape[0]),
        n_pocket_atoms=int(Xp.shape[0]),
        n_ligand_atoms=int(Xlig.shape[0]),
        pocket_cutoff_A=float(pocket_cutoff_A),
        pocket_knn=int(pocket_knn),
        cross_knn=int(cross_knn),
    )
