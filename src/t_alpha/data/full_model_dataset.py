import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class MetaModelDataset(Dataset):
    """
    A custom PyTorch Dataset for loading protein-ligand binding data stored in HDF5 files.

    This dataset class handles loading and preprocessing of graph-based features,
    protein-ligand coordinates, embeddings, and target values for model training
    and evaluation in binding affinity prediction tasks.

    Attributes:
        data_h5_file (str): Path to the HDF5 file containing protein-ligand datasets.
        device (str): The device ('cpu' or 'cuda') where tensors will be loaded.

    Methods:
        __len__(): Returns the total number of data points in the HDF5 file.
        __getitem__(idx): Retrieves and processes a single data point from the HDF5 file.
    """

    def __init__(self, data_h5_file: str, device: str):
        super(MetaModelDataset, self).__init__()
        with h5py.File(data_h5_file, "r") as file:
            self.keys = list(file.keys())

        self.data_h5_file = data_h5_file
        self.device = device

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> Data:
        with h5py.File(self.data_h5_file, "r") as f:
            group = self.keys[idx]
            atom_coords = f[group]["protein_nodes_withH_coords"][:]
            atom_types = f[group]["protein_nodes_withH_atom_types"][:]
            atom_features = f[group]["protein_nodes_withH_features"][:]
            atom_features = np.concatenate([atom_types, atom_features], axis=1)
            ligand_coords = f[group]["ligand_node_coords"][:]
            surface_coords = f[group]["protein_surface_coords"][:]
            surface_normals = f[group]["protein_surface_normals"][:]
            protein_features = f[group]["protein_node_features"][:]
            protein_coords = f[group]["protein_node_coords"][:]
            protein_edge_idx = f[group]["protein_edge_idx"][:]
            protein_edge_attr = f[group]["protein_edge_attrs"][:]
            esm_vector = f[group]["esm2_embedding"][:]
            rdkit_vector = f[group]["rdkit_vector"][:]
            roberta_vector = f[group]["RoBERTa_vector"][:]
            ligand_features = f[group]["ligand_node_features"][:]
            lig_edges = f[group]["ligand_edge_idx"][:]
            lig_attrs = f[group]["ligand_edge_attrs"][:]
            complex_coords = f[group]["complex_node_coords"][:]
            complex_features = f[group]["complex_node_features"][:]
            complex_edges = f[group]["complex_edge_idx"][:]
            complex_edge_attrs = f[group]["complex_edge_attrs"][:]
            operator = f[group]["operator"][()]
            label = f[group]["pKi_value"][()]

            data = Data(
                atom_coords=torch.tensor(atom_coords, dtype=torch.float32).to(
                    self.device
                ),
                atom_features=torch.tensor(atom_features, dtype=torch.float32).to(
                    self.device
                ),
                ligand_coords=torch.tensor(ligand_coords, dtype=torch.float32).to(
                    self.device
                ),
                surface_coords=torch.tensor(surface_coords, dtype=torch.float32).to(
                    self.device
                ),
                surface_normals=torch.tensor(surface_normals, dtype=torch.float32).to(
                    self.device
                ),
                protein_coords=torch.tensor(protein_coords, dtype=torch.float).to(
                    self.device
                ),
                node_features=torch.tensor(protein_features, dtype=torch.float).to(
                    self.device
                ),
                edge_index=torch.tensor(protein_edge_idx, dtype=torch.long)
                .t()
                .contiguous()
                .to(self.device),
                edge_attr=torch.tensor(protein_edge_attr, dtype=torch.float).to(
                    self.device
                ),
                esm_vector=torch.tensor(esm_vector, dtype=torch.float).to(self.device),
                rdkit_vector=torch.tensor(rdkit_vector, dtype=torch.float).to(
                    self.device
                ),
                roberta_vector=torch.tensor(roberta_vector, dtype=torch.float).to(
                    self.device
                ),
                ligand_features=torch.tensor(ligand_features, dtype=torch.float).to(
                    self.device
                ),
                ligand_edges=torch.tensor(lig_edges, dtype=torch.long)
                .t()
                .contiguous()
                .to(self.device),
                ligand_edge_attr=torch.tensor(lig_attrs, dtype=torch.float).to(
                    self.device
                ),
                complex_coords=torch.tensor(complex_coords, dtype=torch.float).to(
                    self.device
                ),
                complex_features=torch.tensor(complex_features, dtype=torch.float).to(
                    self.device
                ),
                complex_edges=torch.tensor(complex_edges, dtype=torch.long)
                .t()
                .contiguous()
                .to(self.device),
                complex_edge_attr=torch.tensor(
                    complex_edge_attrs, dtype=torch.float
                ).to(self.device),
                pdbid=group,
                operator=operator,
                label=torch.tensor(label, dtype=torch.float).to(self.device),
            )

            return data
