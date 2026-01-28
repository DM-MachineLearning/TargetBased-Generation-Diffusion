# Run PLINDER End-to-End Debug Pipeline

This script discovers systems in a PLINDER shard, preprocesses them, trains for 1 epoch, and runs sampling.

## Quick Start

Default (si shard, 20 systems on CPU):

```bash
python scripts/run_plinder_e2e.py
```

With GPU and 50 systems:

```bash
python scripts/run_plinder_e2e.py --take_n 50 --device cuda
```

If your ligands are large (>128 atoms), increase the limit:

```bash
python scripts/run_plinder_e2e.py --take_n 50 --max_ligand_atoms 256 --device cuda
```

Different shard:

```bash
python scripts/run_plinder_e2e.py --shard ab --take_n 30 --device cuda
```

## Flags

-   `--shard`: shard name under `systems/` (default `si`)
-   `--plinder_root`: root with `systems/` (default `data/plinder/2024-06/v2`)
-   `--processed_root`: where processed artifacts are written (default `data/processed/plinder_debug`)
-   `--take_n`: how many systems to preprocess (default 20)
-   `--device`: `cpu` or `cuda` (default `cpu`)
-   `--epochs`: training epochs (default 1)
-   `--batch_size`: training batch size (default 2)
-   `--max_ligand_atoms`: max atoms per ligand (default 128)
-   `--max_pocket_atoms`: max atoms per pocket (default 1024)
-   `--max_edges`: max edges per graph (default 1024)
-   `--run_name`: run directory name (default `plinder_debug`)

## Outputs

-   **Processed artifacts**: `data/processed/plinder_debug/<system_id>/`
    -   `ligand_graph.pt` - ligand structure & bonds
    -   `pocket_graph.pt` - pocket atoms & features
    -   `cross_edges.pt` - ligand-pocket interactions
    -   `meta.json` - preprocessing metadata
-   **Training checkpoint**: `runs/plinder_debug/ckpt.pt`
-   **Samples**: `runs/plinder_debug/samples.pt`
-   **Failures log**: `data/processed/plinder_debug/failures.json` (if any)

## Pipeline Steps

1.  **Discovery**: Loads specified shard (e.g., `data/plinder/2024-06/v2/systems/si`)
2.  **Preprocessing**: Converts each system to graph tensors
3.  **Sanity Check**: Verifies one processed system has valid shapes
4.  **Training**: Runs diffusion model for N epochs
5.  **Sampling**: Generates N samples using the trained model