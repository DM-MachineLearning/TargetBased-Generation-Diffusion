
import torch

class ContinuousDiffusion:
    """DDPM continuous diffusion for coordinates with optional regional noising."""
    def __init__(self, beta: torch.Tensor):
        self.beta = beta
        self.T = int(beta.numel())
        self.alpha = 1.0 - beta
        self.ab = torch.cumprod(self.alpha, 0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, edit_mask: torch.Tensor) -> torch.Tensor:
        """Forward sample with regional noise: only editable atoms receive noise.
        x0: [B,N,3], noise: [B,N,3], edit_mask: [B,N] (1 editable, 0 retained)
        t: [B] int
        """
        abt = self.ab[t].view(-1, 1, 1)
        return abt.sqrt() * x0 + (1 - abt).sqrt() * (noise * edit_mask[:, :, None])

    def p_mean_from_eps(self, x_t: torch.Tensor, eps_hat: torch.Tensor, t: int) -> torch.Tensor:
        """Compute DDPM mean μθ(x_t,t) given predicted eps."""
        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        ab_t = self.ab[t]
        coef = beta_t / (1 - ab_t).sqrt().clamp_min(1e-12)
        mu = (1 / alpha_t.sqrt()) * (x_t - coef * eps_hat)
        return mu

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, eps_hat: torch.Tensor, t: int) -> torch.Tensor:
        """One reverse step for coordinates."""
        mu = self.p_mean_from_eps(x_t, eps_hat, t)
        if t == 0:
            return mu
        sigma = self.beta[t].sqrt()
        return mu + sigma * torch.randn_like(x_t)

def inpaint(x: torch.Tensor, x_ref: torch.Tensor, retain_mask: torch.Tensor) -> torch.Tensor:
    """Hard inpainting: overwrite retained atoms from reference.
    retain_mask: [B,N] (1 retain, 0 editable)
    """
    return x * (1 - retain_mask[:, :, None]) + x_ref * retain_mask[:, :, None]
