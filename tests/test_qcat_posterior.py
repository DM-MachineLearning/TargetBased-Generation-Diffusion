
import torch
from src.diffusion.categorical import QCat
from src.diffusion.schedules import linear_beta

def test_posterior_shapes():
    beta = linear_beta(10, 1e-4, 0.02, device="cpu")
    qc = QCat(5, beta)
    a0 = torch.randint(0,5,(2,7))
    at = qc.q_sample(a0, 5)
    post = qc.posterior(at, a0, 5)
    assert post.shape == (2,7,5)
    assert torch.allclose(post.sum(-1), torch.ones(2,7), atol=1e-4)
