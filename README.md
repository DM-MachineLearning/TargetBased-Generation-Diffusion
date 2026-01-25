
# TaPR-Diff (Target-aware Pharmacophore Retention Diffusion)

This repository contains:
- Continuous DDPM for ligand coordinates with **region-dependent noising**
- **Categorical diffusion (Q-cat / D3PM-style)** for atom and bond types
- **Hard inpainting** for retained atoms during sampling
- A synthetic end-to-end demo
- PLINDER download + preprocessing into **all-atom pocket graphs** and ligand graphs
- Training + sampling scripts for the processed PLINDER graphs

## 1) Create environment (recommended)
```bash
conda env create -f environment.yml
conda activate taprdiff
```

## 2) Download a small slice of PLINDER
```bash
python scripts/download_plinder.py --release 2024-06 --iteration v2 --out data/plinder --prefix splits
python scripts/download_plinder.py --release 2024-06 --iteration v2 --out data/plinder --prefix systems/1a --max_files 200
```

## 3) Preprocess systems into graphs
You must pass the *system IDs* (folder names under systems/*/*).
```bash
python scripts/preprocess_plinder.py --config configs/plinder_small.yaml --system_ids <SYSTEM1> <SYSTEM2>
```

Outputs:
`data/processed/plinder_small/<SYSTEM>/ligand_graph.pt`
`data/processed/plinder_small/<SYSTEM>/pocket_graph.pt`

## 4) Train diffusion model on processed PLINDER graphs
```bash
python scripts/train_plinder.py --config configs/train_plinder_small.yaml
```

## 5) Sample / edit ligands with inpainting
```bash
python scripts/sample_plinder.py --config configs/train_plinder_small.yaml --ckpt runs/train_plinder_small/ckpt.pt --out samples.pt --n 4
```

## Notes
- Current retention mask is a *debug default* (first third of ligand atoms retained). Replace with pharmacophore/scaffold masks from your Step-2 module.
- Ligand parsing requires RDKit; mmCIF parsing uses gemmi.
- This code avoids PyTorch Geometric to keep installation simple; message passing uses torch index_add_.
