
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def time_embedding(t: torch.Tensor, dim: int):
    # sinusoidal, t: [B] float
    device = t.device
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=device).float() * (torch.log(torch.tensor(10000.0, device=device)) / (half-1)))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0],1), device=device)], dim=-1)
    return emb

class PocketConditionedDenoiser(nn.Module):
    """A lightweight but practical pocket-conditioned denoiser (no PyG).

    Inputs (padded):
      X_t: [B,Nl,3]
      A_t: [B,Nl]    (categorical)
      bond edges: src/dst [B,E], bond types B_t [B,E]
      pocket: Xp [B,Np,3], Hp [B,Np,Dp]
      masks: lig_mask [B,Nl], pocket_mask [B,Np], edge_mask [B,E]
      timestep t: int or [B]

    Outputs:
      eps_hat: [B,Nl,3]
      logits_A0: [B,Nl,Ka]
      logits_B0: [B,E,Kb]
    """
    def __init__(self, Ka: int, Kb: int, Dp: int, hidden: int = 256, k_cross: int = 32, layers: int = 4):
        super().__init__()
        self.Ka = Ka
        self.Kb = Kb
        self.hidden = hidden
        self.k_cross = k_cross
        self.layers = layers

        self.atom_emb = nn.Embedding(Ka, hidden)
        self.bond_emb = nn.Embedding(Kb, hidden)

        self.pocket_proj = nn.Linear(Dp, hidden)
        self.coord_proj = nn.Linear(3, hidden)

        self.t_proj = nn.Linear(hidden, hidden)

        self.lig_update = nn.ModuleList([nn.GRUCell(hidden, hidden) for _ in range(layers)])
        self.msg_mlp = nn.ModuleList([nn.Sequential(nn.Linear(hidden*3+4, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.cross_mlp = nn.ModuleList([nn.Sequential(nn.Linear(hidden*2+4, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(layers)])

        self.out_eps = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.out_A = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, Ka))
        self.out_B = nn.Sequential(nn.Linear(hidden*3+4, hidden), nn.ReLU(), nn.Linear(hidden, Kb))

    def _edge_geom(self, Xsrc, Xdst, src, dst):
        # src/dst: [B,E] indices
        B,E = src.shape
        src_x = Xsrc.gather(1, src[:,:,None].expand(-1,-1,3))
        dst_x = Xdst.gather(1, dst[:,:,None].expand(-1,-1,3))
        rel = dst_x - src_x
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)
        rel_unit = rel / dist
        feat = torch.cat([dist, rel_unit], dim=-1)  # [B,E,4]
        return feat, src_x, dst_x

    def _scatter_add(self, msg, index, N):
        # msg: [B,E,H], index: [B,E] -> out [B,N,H]
        B,E,H = msg.shape
        out = torch.zeros((B,N,H), device=msg.device, dtype=msg.dtype)
        for b in range(B):
            out[b].index_add_(0, index[b], msg[b])
        return out

    def _cross_knn(self, Xl, Xp, lig_mask, pocket_mask, k):
        # Compute kNN pocket indices for each ligand atom (masked)
        # Xl [B,Nl,3], Xp [B,Np,3]
        B,Nl,_ = Xl.shape
        Np = Xp.shape[1]
        # large value for masked pockets
        d = torch.cdist(Xl, Xp)  # [B,Nl,Np]
        # mask pockets
        d = d + (1.0 - pocket_mask)[:,None,:] * 1e6
        k_eff = min(k, Np)
        knn = torch.topk(d, k_eff, largest=False).indices  # [B,Nl,k]
        return knn

    def forward(self, X_t, A_t, bond_src, bond_dst, B_t, Xp, Hp, lig_mask, pocket_mask, edge_mask, t):
        B,Nl,_ = X_t.shape
        E = bond_src.shape[1]
        device = X_t.device

        if isinstance(t, int):
            t = torch.full((B,), float(t), device=device)
        else:
            t = t.float()

        ht = time_embedding(t / max(1.0, t.max().item()), self.hidden)
        ht = self.t_proj(ht)  # [B,H]

        hL = self.atom_emb(A_t.clamp(0, self.Ka-1)) + self.coord_proj(X_t)
        hP = self.pocket_proj(Hp)  # [B,Np,H]

        # add time embedding to ligand tokens
        hL = hL + ht[:,None,:]

        for layer in range(self.layers):
            # Intra-ligand messages over bond edges
            geom, _, _ = self._edge_geom(X_t, X_t, bond_src, bond_dst)  # [B,E,4]
            h_src = hL.gather(1, bond_src[:,:,None].expand(-1,-1,self.hidden))
            h_dst = hL.gather(1, bond_dst[:,:,None].expand(-1,-1,self.hidden))
            hb = self.bond_emb(B_t.clamp(0, self.Kb-1))
            msg_in = torch.cat([h_src, h_dst, hb, geom], dim=-1)  # [B,E,3H+4]
            msg = self.msg_mlp[layer](msg_in) * edge_mask[:,:,None]
            agg = self._scatter_add(msg, bond_dst, Nl)  # aggregate to dst
            h_new = self.lig_update[layer](agg.reshape(-1,self.hidden), hL.reshape(-1,self.hidden)).view(B,Nl,self.hidden)

            # Cross messages ligand <- pocket via kNN
            knn = self._cross_knn(X_t, Xp, lig_mask, pocket_mask, self.k_cross)  # [B,Nl,k]
            # gather pocket embeddings and coords
            hPk = hP.gather(1, knn[:,:, :, None].expand(-1,-1,-1,self.hidden))  # [B,Nl,k,H]
            xPk = Xp.gather(1, knn[:,:,:,None].expand(-1,-1,-1,3))              # [B,Nl,k,3]
            xL  = X_t[:,:,None,:].expand(-1,-1,knn.shape[2],-1)
            rel = xPk - xL
            dist = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)
            rel_unit = rel / dist
            geom_cross = torch.cat([dist, rel_unit], dim=-1)  # [B,Nl,k,4]
            hLq = h_new[:,:,None,:].expand_as(hPk)
            cross_in = torch.cat([hLq, hPk, geom_cross], dim=-1)  # [B,Nl,k,2H+4]
            cross_msg = self.cross_mlp[layer](cross_in)  # [B,Nl,k,H]
            # distance-based weights
            w = (1.0 / dist).clamp_max(10.0)
            cross_agg = (cross_msg * w).sum(dim=2)  # [B,Nl,H]
            hL = h_new + cross_agg
            # apply ligand mask
            hL = hL * lig_mask[:,:,None]

        eps_hat = self.out_eps(hL)
        logits_A0 = self.out_A(hL)

        # bond logits conditioned on endpoints
        geomE, _, _ = self._edge_geom(X_t, X_t, bond_src, bond_dst)
        h_src = hL.gather(1, bond_src[:,:,None].expand(-1,-1,self.hidden))
        h_dst = hL.gather(1, bond_dst[:,:,None].expand(-1,-1,self.hidden))
        hb = self.bond_emb(B_t.clamp(0,self.Kb-1))
        e_in = torch.cat([h_src, h_dst, hb, geomE], dim=-1)  # [B,E,3H+4]
        logits_B0 = self.out_B(e_in)

        return eps_hat, logits_A0, logits_B0
