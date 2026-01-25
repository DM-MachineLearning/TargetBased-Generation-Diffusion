
"""Convert PLINDER systems into graph tensors for TaPR-Diff.

This script expects PLINDER to be downloaded under:
  <plinder_root>/systems/<2-char>/<system_id>/...

PLINDER contains standardized files per system. In practice, file names can vary by release.
This preprocessor tries common filenames and provides clear errors when missing.

Outputs per system:
  - pocket_graph.pt: dict with Xp, Hp, edge_index_p, edge_attr_p
  - ligand_graph.pt: dict with X, A, edge_index_l, bond_types (optional)
  - cross_edges.pt: dict with edge_index_lp, edge_attr_lp
  - meta.json: ids, file paths, counts

Run:
  python scripts/preprocess_plinder.py --config configs/plinder_small.yaml --system_ids <id1> <id2> ...

Tip:
  Start by downloading splits + one small systems bucket (e.g., systems/1a) then preprocess those IDs.
"""

import argparse, os, json, glob
from pathlib import Path
import yaml
from tqdm import tqdm

from src.data.plinder.preprocess import system_to_graph

def find_system_dir(plinder_root: str, system_id: str) -> Path:
    # PLINDER groups systems by two-char prefix. We search to be robust.
    root = Path(plinder_root) / "systems"
    # Fast-path: prefix directory by first two chars of system_id if it looks like pdb-ish.
    candidates = []
    if len(system_id) >= 2:
        candidates.append(root / system_id[:2].lower() / system_id)
        candidates.append(root / system_id[:2].upper() / system_id)
    # Fallback glob
    candidates += [Path(p) for p in glob.glob(str(root / "*" / system_id))]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not locate system directory for {system_id} under {root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--system_ids", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0, help="Limit number of systems processed.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    plinder_root = cfg["plinder"]["root"]
    out_dir = Path(cfg["preprocess"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    system_ids = list(args.system_ids)

    # Optional: read from file if provided
    list_path = cfg["plinder"].get("system_list_path") or ""
    if list_path and Path(list_path).exists():
        with open(list_path) as f:
            for line in f:
                s=line.strip()
                if s: system_ids.append(s)

    if not system_ids:
        raise SystemExit("No system IDs provided. Use --system_ids or set plinder.system_list_path in config.")

    if args.limit and args.limit > 0:
        system_ids = system_ids[: args.limit]

    failures = []
    for sid in tqdm(system_ids, desc="Preprocessing systems"):
        try:
            sdir = find_system_dir(plinder_root, sid)
            out_sys = out_dir / sid
            out_sys.mkdir(parents=True, exist_ok=True)
            artifacts = system_to_graph(
                system_dir=sdir,
                out_dir=out_sys,
                pocket_cutoff_A=float(cfg["preprocess"]["pocket_cutoff_A"]),
                pocket_knn=int(cfg["preprocess"]["pocket_knn"]),
                cross_knn=int(cfg["preprocess"]["cross_knn"]),
                include_hydrogens=bool(cfg["preprocess"]["include_hydrogens"]),
                max_pocket_atoms=int(cfg["preprocess"]["max_pocket_atoms"]),
                max_ligand_atoms=int(cfg["preprocess"]["max_ligand_atoms"]),
            )
            with open(out_sys / "meta.json", "w") as f:
                json.dump(artifacts, f, indent=2)
        except Exception as e:
            failures.append((sid, repr(e)))

    if failures:
        fail_path = out_dir / "failures.json"
        with open(fail_path, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"Completed with {len(failures)} failures. See {fail_path}")
    else:
        print("Completed successfully.")

if __name__ == "__main__":
    main()
