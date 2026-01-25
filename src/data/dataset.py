
def collate(batch):
    import torch
    x=torch.stack([b['x'] for b in batch])
    a=torch.stack([b['a'] for b in batch])
    m=torch.stack([b['mask'] for b in batch])
    return x,a,m
