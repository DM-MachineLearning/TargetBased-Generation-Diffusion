import os

# =============================================================================
# PLINDER CONFIGURATION
# These environment variables must be set BEFORE importing plinder.core
# "v2" is the full 2024-06 release. Use "tutorial" only for small-scale testing.
# =============================================================================
os.environ["PLINDER_RELEASE"] = "2024-06"
os.environ["PLINDER_ITERATION"] = "v2" 

import torch
import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem

# Official PLINDER Imports
from plinder.core import PlinderSystem
from plinder.core.scores import query_index

# PharmaDiff Imports (Ensure pharmadiff is in your PYTHONPATH)
from pharmadiff.datasets.dataset_utils import mol_to_torch_geometric
from pharmadiff.datasets.pharmacophore_utils import mol_to_torch_pharmacophore

# Atom encoding for PharmaDiff (Must match your model config)
ATOM_ENCODER = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 8, 'S': 9, 'Cl': 10, 'Br': 12, 'I': 13}

class PlinderGraphDataset(Dataset):
    def __init__(self, split='train', pocket_radius=10.0, transform=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'
            pocket_radius (float): Radius in Angstroms to crop protein context
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split = split
        self.pocket_radius = pocket_radius
        self.transform = transform

        print(f"--> Loading PLINDER index for split: {split}")
        # Load full index with all columns to ensure we get affinity data
        # columns=None fetches all available metadata
        full_index = query_index(columns=None, splits=["train", "val", "test"])
        
        # Filter by split
        self.index = full_index[full_index['split'] == split].reset_index(drop=True)

        # Filter for systems with valid binding affinity (pKd/pIC50)
        # The column name in 2024-06 release is 'affinity_data.pKd'
        if 'affinity_data.pKd' in self.index.columns:
            initial_len = len(self.index)
            self.index = self.index[self.index['affinity_data.pKd'].notna()].reset_index(drop=True)
            print(f"--> Filtered {initial_len} systems down to {len(self.index)} with valid 'affinity_data.pKd'")
        else:
            print("!! WARNING: 'affinity_data.pKd' column not found. Check index columns.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # 1. Retrieve System ID
        entry = self.index.iloc[idx]
        system_id = entry['system_id']

        try:
            # 2. Load System (Triggers download if not cached)
            # MUST use keyword argument for system_id in v2 API
            ps = PlinderSystem(system_id=system_id)

            # 3. Access Components via the .system property (v2 Structure)
            # PLINDER v2 separates metadata from physical system components
            system_obj = ps.system
            
            if system_obj is None:
                return None

            # Ligands: ps.system.ligands is a list of ligand objects
            ligand_list = system_obj.ligands
            if not ligand_list:
                return None
            
            # Grab the first ligand's RDKit molecule (PharmaDiff assumes single ligand context)
            # The property is .rdkit_mol (wrapper around the actual RDKit object)
            ligand_mol = ligand_list[0].rdkit_mol

            # Receptor: ps.system.receptor is the receptor object
            # .structure gives the Biotite AtomArray
            receptor_struct = system_obj.receptor.structure

            if ligand_mol is None or receptor_struct is None:
                return None

            # 4. Compute Pharmacophore (PharmaDiff Logic)
            pharma_data = mol_to_torch_pharmacophore(ligand_mol)
            if pharma_data is None: return None

            # 5. Convert Ligand to Graph
            ligand_data, _ = mol_to_torch_geometric(
                ligand_mol, 
                ATOM_ENCODER, 
                smiles=Chem.MolToSmiles(ligand_mol)
            )

            # 6. Extract Pocket (Context)
            # Calculate centroid of the ligand
            lig_centroid = ligand_mol.GetConformer().GetPositions().mean(axis=0)
            
            # Get receptor coordinates and filter by radius
            rec_coords = receptor_struct.coord
            dists = np.linalg.norm(rec_coords - lig_centroid, axis=1)
            pocket_mask = dists < self.pocket_radius

            # Create tensors for pocket
            pocket_coords = torch.tensor(rec_coords[pocket_mask], dtype=torch.float32)
            pocket_atoms = receptor_struct.element[pocket_mask]
            pocket_feats = self._encode_protein_atoms(pocket_atoms)

            # 7. Get Affinity Label
            # Ensure we cast to float32 for torch compatibility
            affinity_val = entry['affinity_data.pKd']
            affinity = torch.tensor([affinity_val], dtype=torch.float32)

            return {
                'ligand': ligand_data,
                'pharmacophore': pharma_data,
                'pocket_pos': pocket_coords,
                'pocket_feat': pocket_feats,
                'affinity': affinity
            }

        except Exception as e:
            # Catch download/parsing errors to keep training robust
            # print(f"Skipping failed system {system_id}: {e}")
            return None

    def collate(self, data_list):
        """
        Custom collate to handle variable-sized pockets and PyG Batching
        """
        from torch_geometric.data import Batch

        # Filter Nones
        data_list = [d for d in data_list if d is not None]
        if not data_list:
            return None

        # Batch Ligands
        ligands = [d['ligand'] for d in data_list]
        batched_ligand = Batch.from_data_list(ligands)

        # Batch Pockets (Flattened with index vector)
        pocket_pos_list = []
        pocket_feat_list = []
        pocket_batch_list = []
        affinities = []

        for i, data in enumerate(data_list):
            pos = data['pocket_pos']
            feat = data['pocket_feat']

            pocket_pos_list.append(pos)
            pocket_feat_list.append(feat)
            # Create batch index vector: [0, 0, ..., 1, 1, ...]
            pocket_batch_list.append(torch.full((pos.shape[0],), i, dtype=torch.long))
            affinities.append(data['affinity'])

        return {
            'ligand': batched_ligand,
            'pocket_pos': torch.cat(pocket_pos_list, dim=0),
            'pocket_feat': torch.cat(pocket_feat_list, dim=0),
            'pocket_batch': torch.cat(pocket_batch_list, dim=0),
            'affinity': torch.cat(affinities, dim=0)
        }

    def _encode_protein_atoms(self, atoms):
        # Basic One-Hot encoding for protein residues/atoms
        mapping = {'C': 0, 'N': 1, 'O': 2, 'S': 3}
        feats = torch.zeros((len(atoms), 5)) # 5th slot is "Other"
        for i, atom in enumerate(atoms):
            idx = mapping.get(atom, 4)
            feats[i, idx] = 1.0
        return feats

