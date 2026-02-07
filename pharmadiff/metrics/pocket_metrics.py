import torch


def _safe_tanimoto(a: torch.Tensor, b: torch.Tensor) -> float:
    intersection = torch.sum(a * b).item()
    denom = torch.sum(a).item() + torch.sum(b).item() - intersection
    if denom <= 0:
        return 0.0
    return float(intersection / denom)


def compute_contact_map(ligand_pos: torch.Tensor, pocket_pos: torch.Tensor, cutoff: float) -> torch.Tensor:
    if ligand_pos is None or pocket_pos is None:
        return None
    if ligand_pos.numel() == 0 or pocket_pos.numel() == 0:
        return None
    dists = torch.cdist(ligand_pos, pocket_pos)
    return (dists <= cutoff).any(dim=0).to(torch.float32)


def compute_contact_map_satisfaction(
    ref_contacts: torch.Tensor,
    gen_contacts: torch.Tensor,
) -> float:
    if ref_contacts is None or gen_contacts is None:
        return None
    ref_sum = torch.sum(ref_contacts).item()
    if ref_sum <= 0:
        return None
    satisfied = torch.sum(ref_contacts * gen_contacts).item()
    return float(satisfied / ref_sum)


def _pocket_types_from_feat(pocket_feat: torch.Tensor) -> torch.Tensor:
    if pocket_feat is None or pocket_feat.numel() == 0:
        return None
    if pocket_feat.dim() != 2:
        return None
    num_types = min(5, pocket_feat.size(1))
    return torch.argmax(pocket_feat[:, :num_types], dim=-1)


def compute_interaction_fingerprint(
    ligand_pos: torch.Tensor,
    ligand_atom_types: torch.Tensor,
    pocket_pos: torch.Tensor,
    pocket_feat: torch.Tensor,
    cutoff: float,
    num_ligand_types: int,
) -> torch.Tensor:
    if ligand_pos is None or pocket_pos is None:
        return None
    if ligand_pos.numel() == 0 or pocket_pos.numel() == 0:
        return None
    pocket_types = _pocket_types_from_feat(pocket_feat)
    if pocket_types is None:
        pocket_types = torch.zeros(pocket_pos.size(0), dtype=torch.long, device=ligand_pos.device)
        num_pocket_types = 1
    else:
        num_pocket_types = int(torch.max(pocket_types).item()) + 1

    if ligand_atom_types is None or ligand_atom_types.numel() == 0:
        return None
    if ligand_atom_types.dim() != 1:
        ligand_atom_types = ligand_atom_types.view(-1)

    dists = torch.cdist(ligand_pos, pocket_pos)
    contact_indices = torch.nonzero(dists <= cutoff, as_tuple=False)
    fingerprint = torch.zeros(
        num_ligand_types * num_pocket_types,
        dtype=torch.float32,
        device=ligand_pos.device,
    )
    if contact_indices.numel() == 0:
        return fingerprint
    for lig_idx, pocket_idx in contact_indices:
        lig_type = int(ligand_atom_types[lig_idx].item())
        pocket_type = int(pocket_types[pocket_idx].item())
        fingerprint[lig_type * num_pocket_types + pocket_type] = 1.0
    return fingerprint


def compute_ifp_similarity(
    ref_fingerprint: torch.Tensor,
    gen_fingerprint: torch.Tensor,
) -> float:
    if ref_fingerprint is None or gen_fingerprint is None:
        return None
    return _safe_tanimoto(ref_fingerprint, gen_fingerprint)