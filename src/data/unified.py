
import torch
from .synthetic import Synthetic
from .dataset import collate
from .plinder.dataset import ProcessedPlinderDataset

def get_dataset(cfg):
    if cfg['data']['name'] == 'synthetic':
        ds = Synthetic(
            cfg['data']['n_samples'],
            cfg['data']['n_atoms'],
            cfg['data']['n_pocket'],
            cfg['data']['atom_types'],
        )
        return ds, collate

    if cfg['data']['name'] == 'plinder':
        ds = ProcessedPlinderDataset(cfg['data']['root'])
        def collate_plinder(batch):
            # batch_size=1 recommended initially
            b = batch[0]
            lig = b['ligand']
            x = lig['X'].unsqueeze(0).float()
            a = lig['A'].unsqueeze(0).long()
            # default: retain first 1/3 atoms as anchors
            mask = torch.ones_like(a).float()
            mask[:, : max(1, a.size(1)//3)] = 0.0
            return x, a, mask
        return ds, collate_plinder

    raise ValueError(f"Unknown dataset {cfg['data']['name']}")
