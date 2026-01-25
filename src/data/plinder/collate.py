
from __future__ import annotations
import torch

def pad_1d(x, L, pad_value=0):
    if x.numel() >= L:
        return x[:L]
    out = torch.full((L,), pad_value, dtype=x.dtype)
    out[:x.numel()] = x
    return out

def pad_2d(x, L, pad_value=0):
    # x: [N,D] -> [L,D]
    N,D = x.shape
    if N >= L:
        return x[:L]
    out = torch.full((L,D), float(pad_value), dtype=x.dtype)
    out[:N] = x
    return out

def collate_plinder(batch, max_ligand_atoms=256, max_pocket_atoms=2048, max_edges=2048, atom_vocab_size=64, bond_vocab_size=8):
    """Pads variable-size graphs into dense batch tensors.

    Returns:
      X0: [B,Nl,3]
      A0: [B,Nl]
      retain_mask: [B,Nl]  (here default: retain first third; replace later with pharmacophore mask)
      edit_mask: [B,Nl]
      bond_src: [B,E], bond_dst: [B,E], B0: [B,E], edge_mask: [B,E]
      Xp: [B,Np,3], Hp: [B,Np,Dp], pocket_mask: [B,Np]
    """
    B = len(batch)
    # Determine feature dims
    Dp = batch[0]["pocket"]["Hp"].shape[1]

    X0 = torch.zeros((B, max_ligand_atoms, 3), dtype=torch.float32)
    A0 = torch.zeros((B, max_ligand_atoms), dtype=torch.long)
    lig_mask = torch.zeros((B, max_ligand_atoms), dtype=torch.float32)

    # simple default retention: first third retained (team will replace with pharmacophore/scaffold mask)
    retain_mask = torch.zeros((B, max_ligand_atoms), dtype=torch.float32)

    bond_src = torch.zeros((B, max_edges), dtype=torch.long)
    bond_dst = torch.zeros((B, max_edges), dtype=torch.long)
    B0 = torch.zeros((B, max_edges), dtype=torch.long)
    edge_mask = torch.zeros((B, max_edges), dtype=torch.float32)

    Xp = torch.zeros((B, max_pocket_atoms, 3), dtype=torch.float32)
    Hp = torch.zeros((B, max_pocket_atoms, Dp), dtype=torch.float32)
    pocket_mask = torch.zeros((B, max_pocket_atoms), dtype=torch.float32)

    system_id = []

    for i, item in enumerate(batch):
        lig = item["ligand"]
        poc = item["pocket"]

        xl = lig["X"].float()
        al = lig["A"].long()
        nl = min(xl.shape[0], max_ligand_atoms)

        X0[i, :nl] = xl[:nl]
        # clamp atom types into vocab
        A0[i, :nl] = torch.clamp(al[:nl], 0, atom_vocab_size-1)
        lig_mask[i, :nl] = 1.0
        retain_mask[i, :max(1, nl//3)] = 1.0

        # bonds
        ei = lig["edge_index"].long()
        bt = lig["bond_type"].long()
        # edge_index stored directed; we keep as is
        E = min(ei.shape[1], max_edges)
        bond_src[i, :E] = torch.clamp(ei[0, :E], 0, max_ligand_atoms-1)
        bond_dst[i, :E] = torch.clamp(ei[1, :E], 0, max_ligand_atoms-1)
        B0[i, :E] = torch.clamp(bt[:E], 0, bond_vocab_size-1)
        edge_mask[i, :E] = 1.0

        xp = poc["Xp"].float()
        hp = poc["Hp"].float()
        np_ = min(xp.shape[0], max_pocket_atoms)
        Xp[i, :np_] = xp[:np_]
        Hp[i, :np_] = hp[:np_]
        pocket_mask[i, :np_] = 1.0

        system_id.append(item.get("system_id",""))

    edit_mask = lig_mask * (1.0 - retain_mask)

    return {
        "X0": X0,
        "A0": A0,
        "lig_mask": lig_mask,
        "retain_mask": retain_mask,
        "edit_mask": edit_mask,
        "bond_src": bond_src,
        "bond_dst": bond_dst,
        "B0": B0,
        "edge_mask": edge_mask,
        "Xp": Xp,
        "Hp": Hp,
        "pocket_mask": pocket_mask,
        "system_id": system_id,
    }
