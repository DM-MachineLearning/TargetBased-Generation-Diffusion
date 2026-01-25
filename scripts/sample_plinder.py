
import argparse, yaml, torch
from pathlib import Path

from src.data.plinder.dataset import ProcessedPlinderDataset
from src.data.plinder.collate import collate_plinder
from src.diffusion.schedules import linear_beta, cosine_beta
from src.diffusion.continuous import ContinuousDiffusion, inpaint
from src.diffusion.categorical import QCat
from src.models.pocket_denoiser import PocketConditionedDenoiser

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Training config used to build model/diffusion.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="samples.pt")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["device"])

    ds = ProcessedPlinderDataset(cfg["data"]["processed_root"])
    dl = torch.utils.data.DataLoader(ds, batch_size=args.n, shuffle=False,
        collate_fn=lambda b: collate_plinder(
            b,
            max_ligand_atoms=cfg["data"]["max_ligand_atoms"],
            max_pocket_atoms=cfg["data"]["max_pocket_atoms"],
            max_edges=cfg["data"]["max_edges"],
            atom_vocab_size=cfg["data"]["atom_vocab_size"],
            bond_vocab_size=cfg["data"]["bond_vocab_size"],
        ))

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
    Dp = Hp.shape[-1]
    model = PocketConditionedDenoiser(
        Ka=cfg["data"]["atom_vocab_size"],
        Kb=cfg["data"]["bond_vocab_size"],
        Dp=Dp,
        hidden=cfg["model"]["hidden"],
        k_cross=cfg["model"]["k_cross"],
        layers=cfg["model"]["layers"],
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Initialize x_T by noising editable region only
    Xt = X0.clone()
    At = A0.clone()
    Bt = B0.clone()
    # set At, Bt to random for editable atoms/edges to encourage sampling
    At = torch.where(edit.bool(), torch.randint_like(At, 0, cfg["data"]["atom_vocab_size"]), At)
    Bt = torch.where(edge_mask.bool(), torch.randint_like(Bt, 0, cfg["data"]["bond_vocab_size"]), Bt)
    Xt = Xt + torch.randn_like(Xt) * edit[:,:,None]

    # reverse loop
    for t in reversed(range(T)):
        eps_hat, logits_A0, logits_B0 = model(Xt, At, bond_src, bond_dst, Bt, Xp, Hp, lig_mask, pocket_mask, edge_mask, t)
        Xt = cont.p_sample(Xt, eps_hat, t)
        Xt = inpaint(Xt, X0, retain)  # keep retained anchors fixed
        # categorical reverse steps
        At = qA.p_sample(logits_A0, At, t)
        # inpaint retained atom types to original
        At = torch.where(retain.bool(), A0, At)
        Bt = qB.p_sample(logits_B0, Bt, t)

    torch.save({
        "X_gen": Xt.cpu(),
        "A_gen": At.cpu(),
        "B_gen": Bt.cpu(),
        "X_seed": X0.cpu(),
        "A_seed": A0.cpu(),
        "retain_mask": retain.cpu(),
        "system_id": batch["system_id"],
    }, args.out)
    print("Saved samples to", args.out)

if __name__ == "__main__":
    main()
