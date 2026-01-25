
import torch

class QCat:
    """Categorical diffusion with Q_t transition matrices.

    Forward:
      q(a_t | a_{t-1}) = Cat(Q_t[a_{t-1}])
    where Q_t = (1-β_t) I + β_t U, with U uniform (can be replaced by any base matrix).

    We also cache Q̄_t = Q_1 ... Q_t.

    Posterior for D3PM-style sampling:
      q(a_{t-1}=k | a_t=j, a0=i) ∝ Q_t[k,j] * Q̄_{t-1}[i,k]
    """
    def __init__(self, K: int, beta: torch.Tensor, Q_base: torch.Tensor | None = None):
        self.K = int(K)
        self.T = int(beta.numel())
        self.beta = beta
        device = beta.device

        if Q_base is None:
            Q_base = torch.ones(self.K, self.K, device=device) / self.K
        else:
            assert Q_base.shape == (self.K, self.K)
            Q_base = Q_base.to(device)

        Qs = []
        for t in range(self.T):
            Qt = (1.0 - beta[t]) * torch.eye(self.K, device=device) + beta[t] * Q_base
            Qs.append(Qt)
        self.Q = torch.stack(Qs, dim=0)              # [T,K,K]
        self.Qbar = torch.zeros_like(self.Q)         # [T,K,K]
        self.Qbar[0] = self.Q[0]
        for t in range(1, self.T):
            self.Qbar[t] = self.Qbar[t-1] @ self.Q[t]

    @torch.no_grad()
    def q_sample(self, a0: torch.Tensor, t: int) -> torch.Tensor:
        """Sample a_t given a0 using q(a_t|a0)=Cat(Q̄_t[a0]).
        a0: [B,N] int
        returns: [B,N] int
        """
        probs = self.Qbar[t][a0]  # [B,N,K]
        return torch.multinomial(probs.view(-1, self.K), 1).view(a0.shape)

    def posterior(self, a_t: torch.Tensor, a0: torch.Tensor, t: int) -> torch.Tensor:
        """Compute posterior probs over a_{t-1}:
        q(a_{t-1} | a_t, a0) for each token.
        a_t, a0: [B,N] int
        returns: [B,N,K]
        """
        assert t >= 1, "posterior defined for t>=1"
        # Q_t[k, j] where j = a_t
        Qt = self.Q[t]  # [K,K]
        # Gather column for each a_t: [B,N,K] where along K is k
        col = Qt[:, a_t.reshape(-1)].T  # [B*N, K]
        col = col.view(*a_t.shape, self.K)

        # Qbar_{t-1}[i, k] where i = a0
        Qbar_prev = self.Qbar[t-1]  # [K,K]
        row = Qbar_prev[a0]         # [B,N,K]

        unnorm = col * row
        norm = unnorm.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return unnorm / norm

    @torch.no_grad()
    def p_sample(self, logits_a0: torch.Tensor, a_t: torch.Tensor, t: int) -> torch.Tensor:
        """One reverse categorical step using model logits for a0.

        logits_a0: [B,N,K] predicting p(a0|.)
        a_t: [B,N]
        returns a_{t-1}: [B,N]
        """
        # sample a0-hat (can also use argmax for deterministic)
        probs_a0 = torch.softmax(logits_a0, dim=-1)
        a0_hat = torch.multinomial(probs_a0.view(-1, self.K), 1).view(a_t.shape)

        if t == 0:
            return a0_hat

        post = self.posterior(a_t=a_t, a0=a0_hat, t=t)  # [B,N,K]
        return torch.multinomial(post.view(-1, self.K), 1).view(a_t.shape)
