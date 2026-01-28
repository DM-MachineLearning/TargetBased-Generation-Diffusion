#!/usr/bin/env python3
"""End-to-end PLINDER debug runner.

This script performs a small, self-contained pipeline for debugging:
 - discover a non-empty shard under `<plinder_root>/systems/`
 - pick first `--take_n` system ids and run `system_to_graph` on them
 - run a sanity check on one processed system
 - run a short training loop (default 1 epoch)
 - run sampling for `n` outputs and save samples

This imports the library functions directly (no subprocesses / gsutil).
"""
from __future__ import annotations
import sys
import argparse
import json
import traceback
from pathlib import Path
import os
from typing import List

import torch

# Ensure repo root is in path so src can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.plinder.preprocess import system_to_graph
from src.data.plinder.dataset import ProcessedPlinderDataset
from src.data.plinder.collate import collate_plinder
from src.models.pocket_denoiser import PocketConditionedDenoiser
from src.diffusion.continuous import ContinuousDiffusion, inpaint
from src.diffusion.categorical import QCat
from src.diffusion.schedules import cosine_beta
from src.diffusion.losses import masked_mse, masked_ce, masked_edge_ce
from src.utils.seed import set_seed


def get_shard(plinder_root: Path, shard_name: str) -> Path:
    """Load a specific shard by name."""
    shard = plinder_root / "systems" / shard_name
    if not shard.exists():
        raise FileNotFoundError(f"Shard '{shard_name}' not found at {shard}")
    return shard


def list_system_ids(shard: Path, take_n: int) -> List[str]:
    ids = [p.name for p in sorted(shard.iterdir()) if p.is_dir()]
    return ids[:take_n]


def run_preprocess(plinder_root: Path, processed_root: Path, system_ids: List[str], failures_path: Path, **kwargs):
    failures = []
    processed_root.mkdir(parents=True, exist_ok=True)
    for sid in system_ids:
        system_dir = plinder_root / "systems" / sid
        out_dir = processed_root / sid
        try:
            print(f"Preprocessing {sid} -> {out_dir}")
            meta = system_to_graph(system_dir, out_dir, **kwargs)
            # write meta.json alongside artifacts
            with open(out_dir / "meta.json", "w") as fh:
                json.dump(meta, fh)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Failed {sid}: {e}")
            failures.append({"system_id": sid, "error": str(e), "traceback": tb})
            continue
    # dump failures
    if failures:
        with open(failures_path, "w") as fh:
            json.dump(failures, fh, indent=2)
    return failures


def sanity_check(processed_root: Path):
    ds = ProcessedPlinderDataset(str(processed_root))
    if len(ds) == 0:
        raise SystemExit(f"No processed systems found under {processed_root}")
    sample = ds[0]
    lig = sample["ligand"]
    poc = sample["pocket"]
    cross = sample["cross"]
    print("Sanity check shapes:")
    print(" ligand X:", lig["X"].shape)
    print(" pocket Xp:", poc["Xp"].shape)
    print(" cross edges:", cross["edge_index"].shape)
    assert lig["X"].shape[0] > 0, "Ligand has zero atoms"
    assert poc["Xp"].shape[0] > 0, "Pocket has zero atoms"
    assert cross["edge_index"].numel() > 0, "Cross edges empty"


def build_beta(T: int, device: torch.device):
    return cosine_beta(T, device=device)


def train_short(
    processed_root: Path,
    run_root: Path,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 2,
    max_ligand_atoms: int = 128,
    max_pocket_atoms: int = 1024,
    max_edges: int = 1024,
    atom_vocab_size: int = 64,
    bond_vocab_size: int = 4,
):
    ds = ProcessedPlinderDataset(str(processed_root))
    if len(ds) == 0:
        raise SystemExit(f"No processed systems found under {processed_root}")

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_plinder(
            b,
            max_ligand_atoms=max_ligand_atoms,
            max_pocket_atoms=max_pocket_atoms,
            max_edges=max_edges,
            atom_vocab_size=atom_vocab_size,
            bond_vocab_size=bond_vocab_size,
        ),
    )

    # schedule
    T = 100
    beta = build_beta(T, device=device)
    cont = ContinuousDiffusion(beta)
    qA = QCat(atom_vocab_size, beta)
    qB = QCat(bond_vocab_size, beta)

    # model
    sample0 = ds[0]
    Dp = sample0["pocket"]["Hp"].shape[1]
    model = PocketConditionedDenoiser(Ka=atom_vocab_size, Kb=bond_vocab_size, Dp=Dp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    run_root.mkdir(parents=True, exist_ok=True)

    lam_x = 1.0
    lam_a = 1.0
    lam_b = 1.0

    model.train()
    for ep in range(epochs):
        for batch in dl:
            X0 = batch["X0"].to(device)
            A0 = batch["A0"].to(device)
            B0 = batch["B0"].to(device)
            retain = batch["retain_mask"].to(device)
            edit = batch["edit_mask"].to(device)
            lig_mask = batch["lig_mask"].to(device)
            edge_mask = batch["edge_mask"].to(device)
            bond_src = batch["bond_src"].to(device)
            bond_dst = batch["bond_dst"].to(device)
            Xp = batch["Xp"].to(device)
            Hp = batch["Hp"].to(device)
            pocket_mask = batch["pocket_mask"].to(device)

            Bsz = X0.shape[0]
            t = torch.randint(0, T, (Bsz,), device=device)
            noise = torch.randn_like(X0)
            Xt = cont.q_sample(X0, t, noise, edit)

            At = torch.empty_like(A0)
            Bt = torch.empty_like(B0)
            for i in range(Bsz):
                At[i] = qA.q_sample(A0[i:i+1], int(t[i].item()))[0]
                Bt[i] = qB.q_sample(B0[i:i+1], int(t[i].item()))[0]

            eps_hat, logits_A0, logits_B0 = model(
                Xt, At, bond_src, bond_dst, Bt, Xp, Hp, lig_mask, pocket_mask, edge_mask, t.float()
            )

            loss_x = masked_mse(eps_hat, noise, edit)
            loss_a = masked_ce(logits_A0, A0, edit)
            loss_b = masked_edge_ce(logits_B0, B0, edge_mask)
            loss = lam_x * loss_x + lam_a * loss_a + lam_b * loss_b

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # save final checkpoint
    ckpt = {"model": model.state_dict()}
    torch.save(ckpt, run_root / "ckpt.pt")
    print("Saved checkpoint:", run_root / "ckpt.pt")
    return model, cont, qA, qB


def sample_and_save(model, cont: ContinuousDiffusion, qA: QCat, qB: QCat, processed_root: Path, run_root: Path, device: torch.device, n: int = 4, batch_size: int = 2, max_ligand_atoms: int = 128, max_pocket_atoms: int = 1024, max_edges: int = 1024, atom_vocab_size: int = 64, bond_vocab_size: int = 4):
    ds = ProcessedPlinderDataset(str(processed_root))
    if len(ds) == 0:
        raise SystemExit("No processed data for sampling")

    # create a small batch to obtain shapes; sampling will use these masks and reference retain positions
    B = min(n, len(ds))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=B,
        shuffle=True,
        collate_fn=lambda b: collate_plinder(
            b,
            max_ligand_atoms=max_ligand_atoms,
            max_pocket_atoms=max_pocket_atoms,
            max_edges=max_edges,
            atom_vocab_size=atom_vocab_size,
            bond_vocab_size=bond_vocab_size,
        ),
    )

    batch = next(iter(dl))
    X0 = batch["X0"].to(device)
    A0 = batch["A0"].to(device)
    B0 = batch["B0"].to(device)
    retain = batch["retain_mask"].to(device)
    edit = batch["edit_mask"].to(device)
    lig_mask = batch["lig_mask"].to(device)
    edge_mask = batch["edge_mask"].to(device)
    bond_src = batch["bond_src"].to(device)
    bond_dst = batch["bond_dst"].to(device)
    Xp = batch["Xp"].to(device)
    Hp = batch["Hp"].to(device)
    pocket_mask = batch["pocket_mask"].to(device)

    Bsz = X0.shape[0]
    T = cont.T

    # initialize a_T and b_T and x_T
    x_t = torch.randn_like(X0)
    a_t = torch.empty_like(A0)
    b_t = torch.empty_like(B0)
    for i in range(Bsz):
        a_t[i] = qA.q_sample(A0[i:i+1], T - 1)[0]
        b_t[i] = qB.q_sample(B0[i:i+1], T - 1)[0]

    model.eval()
    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            eps_hat, logits_A0, logits_B0 = model(
                x_t, a_t, bond_src, bond_dst, b_t, Xp, Hp, lig_mask, pocket_mask, edge_mask, torch.full((Bsz,), float(t), device=device)
            )

            # update coordinates
            x_prev = cont.p_sample(x_t, eps_hat, t)
            # inpaint retained atoms
            x_prev = inpaint(x_prev, X0, retain)

            # update categoricals
            a_prev = qA.p_sample(logits_A0, a_t, t)
            b_prev = qB.p_sample(logits_B0, b_t, t)

            x_t = x_prev
            a_t = a_prev
            b_t = b_prev

    samples = {"X": x_t.cpu(), "A": a_t.cpu(), "B": b_t.cpu(), "system_id": batch.get("system_id", [])}
    torch.save(samples, run_root / "samples.pt")
    print("Saved samples:", run_root / "samples.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plinder_root", default="data/plinder/2024-06/v2")
    ap.add_argument("--shard", default="si", help="Shard name under systems/ (e.g., 'si', 'ab', 'k9')")
    ap.add_argument("--processed_root", default="data/processed/plinder_debug")
    ap.add_argument("--take_n", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_ligand_atoms", type=int, default=128)
    ap.add_argument("--max_pocket_atoms", type=int, default=1024)
    ap.add_argument("--max_edges", type=int, default=1024)
    ap.add_argument("--run_name", default="plinder_debug")
    args = ap.parse_args()

    set_seed(42)

    plinder_root = Path(args.plinder_root)
    processed_root = Path(args.processed_root)
    failures_path = processed_root / "failures.json"
    run_root = Path("runs") / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # get shard directly (no discovery)
    shard = get_shard(plinder_root, args.shard)
    print(f"Using shard: {shard}")
    system_ids = list_system_ids(shard, args.take_n)
    print(f"Selected {len(system_ids)} systems (first {min(5, len(system_ids))}):", system_ids[:5])

    # preprocess
    failures = run_preprocess(plinder_root, processed_root, system_ids, failures_path,
                              pocket_cutoff_A=10.0, pocket_knn=24, cross_knn=32,
                              include_hydrogens=False, max_pocket_atoms=args.max_pocket_atoms, max_ligand_atoms=args.max_ligand_atoms)
    if failures:
        print(f"Preprocessing had {len(failures)} failures; details in {failures_path}")

    # sanity check
    sanity_check(processed_root)

    # short training
    model, cont, qA, qB = train_short(
        processed_root,
        run_root,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_ligand_atoms=args.max_ligand_atoms,
        max_pocket_atoms=args.max_pocket_atoms,
        max_edges=args.max_edges,
        atom_vocab_size=64,
        bond_vocab_size=4,
    )

    # sampling
    sample_and_save(model, cont, qA, qB, processed_root, run_root, device, n=4)


if __name__ == "__main__":
    main()
