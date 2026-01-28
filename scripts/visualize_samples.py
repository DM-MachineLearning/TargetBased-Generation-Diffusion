#!/usr/bin/env python3
"""Visualize generated samples from samples.pt and processed systems.

Usage:
  python scripts/visualize_samples.py --samples runs/plinder_debug/samples.pt --output samples_viz/
"""
import sys
import argparse
from pathlib import Path
import json

import torch
import numpy as np

# Try to import visualization tools
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    HAS_RDKIT = True
except:
    HAS_RDKIT = False

# Atom vocab mapping (must match training config)
ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',  # 0-8
    'H', 'C', 'N', 'O', 'S', 'P',                     # 9-14 (heavy atoms, some duplicates for vocab)
    'X', 'X', 'X', 'X', 'X',                           # 15-19 (padding)
] + ['X'] * 44  # Pad to 64 vocab size

# Bond type mapping (1=single, 2=double, 3=triple, 0=none)
BOND_TYPES = {0: Chem.BondType.SINGLE, 1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


def load_samples(pt_file: Path) -> dict:
    """Load samples from .pt file."""
    data = torch.load(pt_file)
    return data


def load_processed_system(system_dir: Path) -> dict:
    """Load preprocessed system graphs."""
    lig = torch.load(system_dir / "ligand_graph.pt")
    poc = torch.load(system_dir / "pocket_graph.pt")
    cross = torch.load(system_dir / "cross_edges.pt")
    meta = json.load(open(system_dir / "meta.json")) if (system_dir / "meta.json").exists() else {}
    return {"ligand": lig, "pocket": poc, "cross": cross, "meta": meta}


def atoms_bonds_to_mol(atom_indices: np.ndarray, coords: np.ndarray, bond_src: np.ndarray, bond_dst: np.ndarray, bond_types: np.ndarray):
    """Convert atom indices, coords, and bonds to RDKit mol."""
    if not HAS_RDKIT:
        print("RDKit not available; skipping molecule creation")
        return None

    # Filter valid atoms (not padding)
    valid = atom_indices > 0
    if not valid.any():
        return None

    valid_idx = np.where(valid)[0]
    atom_map = {old: new for new, old in enumerate(valid_idx)}

    # Create molecule
    mol = Chem.RWMol()
    for idx in valid_idx:
        atom_num = min(int(atom_indices[idx]), len(ATOM_SYMBOLS) - 1)
        symbol = ATOM_SYMBOLS[atom_num]
        if symbol == 'X':
            continue
        try:
            atom = Chem.Atom(symbol)
            mol.AddAtom(atom)
        except:
            pass

    # Add bonds (filter to valid atoms)
    for src, dst, btype in zip(bond_src, bond_dst, bond_types):
        if src in atom_map and dst in atom_map and src < len(atom_indices) and dst < len(atom_indices):
            if valid[src] and valid[dst]:
                src_new = atom_map[src]
                dst_new = atom_map[dst]
                bond_order = BOND_TYPES.get(int(btype), Chem.BondType.SINGLE)
                mol.AddBond(src_new, dst_new, bond_order)

    if mol.GetNumAtoms() == 0:
        return None

    mol = mol.GetMol()

    # Set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, idx in enumerate(valid_idx):
        if i < len(coords):
            conf.SetAtomPosition(i, coords[idx])
    mol.AddConformer(conf, assignId=True)

    return mol


def visualize_samples(samples_file: Path, output_dir: Path = None):
    """Load and display samples."""
    print(f"Loading samples from {samples_file}")
    samples = load_samples(samples_file)

    X = samples["X"].numpy()  # [B, Nl, 3] coordinates
    A = samples["A"].numpy()  # [B, Nl] atom types
    B = samples["B"].numpy()  # [B, E] bond types (if present)
    system_ids = samples.get("system_id", [])

    print(f"Loaded {X.shape[0]} samples")
    print(f"  Shapes: X={X.shape}, A={A.shape}, B={B.shape}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_RDKIT:
        print("\nRDKit not installed. Install with: conda install -c conda-forge rdkit")
        print("For now, showing tensor info:")
        for i in range(min(3, X.shape[0])):
            print(f"\nSample {i}:")
            print(f"  Non-zero atoms: {(A[i] > 0).sum()}")
            print(f"  Coord range: X=[{X[i].min():.2f}, {X[i].max():.2f}]")
        return

    print("\nProcessing samples with RDKit:")
    for i in range(X.shape[0]):
        coords = X[i]  # [Nl, 3]
        atoms = A[i]  # [Nl]

        # For simplicity, assume bond edges are not in samples or use simple heuristic
        # In a real setup, you'd also load bond_src/bond_dst/bond_types
        # Here we'll just create a molecule with atoms and simple connectivity

        mol = Chem.RWMol()
        valid_atoms = []
        for j, atom_idx in enumerate(atoms):
            if atom_idx <= 0 or atom_idx >= len(ATOM_SYMBOLS):
                continue
            symbol = ATOM_SYMBOLS[int(atom_idx)]
            if symbol == 'X':
                continue
            try:
                atom = Chem.Atom(symbol)
                mol.AddAtom(atom)
                valid_atoms.append(j)
            except:
                pass

        if mol.GetNumAtoms() < 2:
            print(f"Sample {i}: Too few atoms ({mol.GetNumAtoms()})")
            continue

        # Add simple connectivity based on distance
        for ii, idx1 in enumerate(valid_atoms):
            for jj, idx2 in enumerate(valid_atoms):
                if ii < jj:
                    dist = np.linalg.norm(coords[idx1] - coords[idx2])
                    if dist < 1.8:  # rough bond distance threshold
                        mol.AddBond(ii, jj, Chem.BondType.SINGLE)

        mol = mol.GetMol()

        # Set conformer
        conf = Chem.Conformer(mol.GetNumAtoms())
        atom_idx = 0
        for idx in valid_atoms:
            if atom_idx < mol.GetNumAtoms():
                pos = coords[idx].astype(float)
                conf.SetAtomPosition(atom_idx, (float(pos[0]), float(pos[1]), float(pos[2])))
                atom_idx += 1
        mol.AddConformer(conf, assignId=True)

        # Sanitize to fix valencies
        try:
            Chem.SanitizeMol(mol)
        except:
            # If sanitization fails, just use it as-is
            pass

        smi = Chem.MolToSmiles(mol)
        try:
            mw = Descriptors.MolWt(mol)
        except:
            mw = 0.0
        sys_id = system_ids[i] if i < len(system_ids) else f"sample_{i}"

        print(f"  Sample {i} ({sys_id}): {mol.GetNumAtoms()} atoms, MW={mw:.1f}, SMILES={smi[:60]}")

        if output_dir:
            # Save as SDF
            sdf_file = output_dir / f"sample_{i:03d}.sdf"
            writer = Chem.SDWriter(str(sdf_file))
            writer.write(mol)
            writer.close()

            # Save SMILES
            with open(output_dir / f"sample_{i:03d}.smi", "w") as f:
                f.write(f"{smi} {sys_id}\n")

    if output_dir:
        print(f"\nSaved structures to {output_dir}/")


def visualize_processed(processed_root: Path):
    """Show info from processed systems."""
    print(f"\nProcessed systems in {processed_root}:")
    for system_dir in sorted(processed_root.iterdir()):
        if not system_dir.is_dir():
            continue
        try:
            data = load_processed_system(system_dir)
            lig = data["ligand"]
            print(f"  {system_dir.name}:")
            print(f"    Ligand atoms: {lig['X'].shape[0]}, bonds: {lig['edge_index'].shape[1]}")
            print(f"    Pocket atoms: {data['pocket']['Xp'].shape[0]}")
        except Exception as e:
            print(f"  {system_dir.name}: Error - {e}")


def main():
    ap = argparse.ArgumentParser(description="Visualize PLINDER samples and processed systems")
    ap.add_argument("--samples", help="Path to samples.pt file")
    ap.add_argument("--processed_root", help="Path to processed systems root")
    ap.add_argument("--output", default="samples_viz", help="Output directory for structures")
    args = ap.parse_args()

    if not args.samples and not args.processed_root:
        print("Usage: python scripts/visualize_samples.py --samples runs/plinder_debug/samples.pt --output samples_viz/")
        print("       or: python scripts/visualize_samples.py --processed_root data/processed/plinder_debug")
        return

    if args.samples:
        samples_file = Path(args.samples)
        if samples_file.exists():
            visualize_samples(samples_file, Path(args.output) if args.output else None)
        else:
            print(f"Samples file not found: {samples_file}")

    if args.processed_root:
        processed_root = Path(args.processed_root)
        if processed_root.exists():
            visualize_processed(processed_root)
        else:
            print(f"Processed root not found: {processed_root}")


if __name__ == "__main__":
    main()
