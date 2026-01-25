
import torch, torch.nn as nn
class Denoiser(nn.Module):
    def __init__(self, K, H):
        super().__init__()
        self.emb=nn.Embedding(K,H)
        self.mlp=nn.Sequential(nn.Linear(H+3,H),nn.ReLU(),nn.Linear(H,H))
        self.out_eps=nn.Linear(H,3)
        self.out_a=nn.Linear(H,K)
    def forward(self, x, a):
        h=self.emb(a)
        h=self.mlp(torch.cat([h,x],-1))
        return self.out_eps(h), self.out_a(h)
