
import argparse, os, yaml, torch
from tqdm import tqdm

from src.utils.seed import set_seed
from src.data.plinder.dataset import ProcessedPlinderDataset
from src.data.plinder.collate import collate_plinder
from src.diffusion.schedules import linear_beta, cosine_beta
from src.diffusion.continuous import ContinuousDiffusion
from src.diffusion.categorical import QCat
from src.diffusion.losses import masked_mse, masked_ce, masked_edge_ce
from src.models.pocket_denoiser import PocketConditionedDenoiser

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    ds = ProcessedPlinderDataset(cfg["data"]["processed_root"])
    if len(ds) == 0:
        raise SystemExit(f"No processed systems found under {cfg['data']['processed_root']}. Run preprocess_plinder.py first.")
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_plinder(
            b,
            max_ligand_atoms=cfg["data"]["max_ligand_atoms"],
            max_pocket_atoms=cfg["data"]["max_pocket_atoms"],
            max_edges=cfg["data"]["max_edges"],
            atom_vocab_size=cfg["data"]["atom_vocab_size"],
            bond_vocab_size=cfg["data"]["bond_vocab_size"],
        ),
    )

    # diffusion schedules
    T = cfg["diffusion"]["T"]
    if cfg["diffusion"].get("schedule","linear") == "cosine":
        beta = cosine_beta(T, device=device)
    else:
        beta = linear_beta(T, cfg["diffusion"]["beta_start"], cfg["diffusion"]["beta_end"], device=device)

    cont = ContinuousDiffusion(beta)
    qA = QCat(cfg["data"]["atom_vocab_size"], beta)
    qB = QCat(cfg["data"]["bond_vocab_size"], beta)

    # model
    # pocket feature dim from one sample
    sample0 = ds[0]
    Dp = sample0["pocket"]["Hp"].shape[1]
    model = PocketConditionedDenoiser(
        Ka=cfg["data"]["atom_vocab_size"],
        Kb=cfg["data"]["bond_vocab_size"],
        Dp=Dp,
        hidden=cfg["model"]["hidden"],
        k_cross=cfg["model"]["k_cross"],
        layers=cfg["model"]["layers"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    os.makedirs(f"runs/{cfg['run_name']}", exist_ok=True)

    lam_x = cfg["train"]["lambda_x"]
    lam_a = cfg["train"]["lambda_a"]
    lam_b = cfg["train"]["lambda_b"]

    model.train()
    for ep in range(cfg["train"]["epochs"]):
        pbar = tqdm(dl, desc=f"epoch {ep}")
        for batch in pbar:
            X0 = batch["X0"].to(device)
            A0 = batch["A0"].to(device)
            B0 = batch["B0"].to(device)
            retain = batch["retain_mask"].to(device)      # 1 retain
            edit = batch["edit_mask"].to(device)          # 1 editable
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

            # categorical forward samples (using q(a_t|a0))
            # For simplicity: use first t in batch for q_sample (can vectorize later); to keep correct, loop.
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
            # bond loss: editable bonds = any bond touching editable atoms; approximate via edge_mask (all) for now
            loss_b = masked_edge_ce(logits_B0, B0, edge_mask)

            loss = lam_x*loss_x + lam_a*loss_a + lam_b*loss_b

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pbar.set_postfix({"loss": float(loss), "x": float(loss_x), "a": float(loss_a), "b": float(loss_b)})

        ckpt = {
            "model": model.state_dict(),
            "cfg": cfg,
        }
        torch.save(ckpt, f"runs/{cfg['run_name']}/ckpt_epoch{ep}.pt")

    torch.save({"model": model.state_dict(), "cfg": cfg}, f"runs/{cfg['run_name']}/ckpt.pt")
    print("Saved:", f"runs/{cfg['run_name']}/ckpt.pt")

if __name__ == "__main__":
    main()
