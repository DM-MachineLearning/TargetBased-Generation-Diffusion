
import torch
import torch.nn.functional as F

def masked_mse(pred, target, mask):
    # pred/target: [B,N,3], mask: [B,N] 1 for editable
    return (((pred - target) ** 2) * mask[:, :, None]).sum() / (mask.sum() * pred.shape[-1] + 1e-12)

def masked_ce(logits, target, mask):
    # logits: [B,N,K], target: [B,N], mask [B,N] (1 editable)
    B,N,K = logits.shape
    logits2 = logits.reshape(B*N, K)
    target2 = target.reshape(B*N)
    mask2 = mask.reshape(B*N).float()
    loss = F.cross_entropy(logits2, target2, reduction="none")
    return (loss * mask2).sum() / (mask2.sum() + 1e-12)

def masked_edge_ce(logits, target, edge_mask):
    # logits: [B,E,K], target: [B,E], edge_mask: [B,E]
    B,E,K = logits.shape
    loss = F.cross_entropy(logits.reshape(B*E, K), target.reshape(B*E), reduction="none")
    m = edge_mask.reshape(B*E).float()
    return (loss * m).sum() / (m.sum() + 1e-12)
