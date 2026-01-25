
"""Dataset loader for preprocessed PLINDER graphs.

Each item reads:
  ligand_graph.pt, pocket_graph.pt, cross_edges.pt, meta.json
and returns a dict suitable for a future full model.

This loader is intentionally minimal; your main model will add bonds, masks, pharmacophores, etc.
"""
from __future__ import annotations
from pathlib import Path
import json, torch

class ProcessedPlinderDataset(torch.utils.data.Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.system_dirs = [p for p in self.root.iterdir() if p.is_dir() and (p / 'ligand_graph.pt').exists()]
        self.system_dirs.sort()

    def __len__(self): return len(self.system_dirs)

    def __getitem__(self, idx):
        d = self.system_dirs[idx]
        lig = torch.load(d / "ligand_graph.pt")
        poc = torch.load(d / "pocket_graph.pt")
        xed = torch.load(d / "cross_edges.pt")
        meta = json.load(open(d / "meta.json")) if (d/"meta.json").exists() else {}
        return {"ligand": lig, "pocket": poc, "cross": xed, "meta": meta, "system_id": d.name}
