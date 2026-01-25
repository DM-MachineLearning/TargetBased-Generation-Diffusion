
import torch, math

def linear_beta(T, b0, b1, device):
    return torch.linspace(b0, b1, T, device=device)

def cosine_beta(T, s=0.008, device="cpu"):
    # Nichol & Dhariwal cosine schedule -> returns betas
    steps = torch.arange(T+1, device=device, dtype=torch.float32)
    f = torch.cos(((steps/T) + s) / (1+s) * math.pi/2) ** 2
    ab = f / f[0]
    betas = 1 - (ab[1:] / ab[:-1]).clamp(0.0001, 0.9999)
    return betas
