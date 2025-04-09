# The class in this file is adapted from the dMaSIF software corresponding
# to this paper: <https://www.biorxiv.org/content/10.1101/2020.12.28.424589v1.full>

import torch.nn as nn

from t_alpha.models.dmasif import dMaSIF
from t_alpha.utils.run import iterate


class SurfaceConvNet(nn.Module):
    def __init__(self, device: str, **kwargs):
        super(SurfaceConvNet, self).__init__()

        # Set default parameters
        self.config = {
            "curvature_scales": [
                1.0,
                2.0,
                3.0,
                5.0,
                10.0,
            ],  # 5 different scales for computing 5 geometric features at each point
            "resolution": 1.0,  # The size of the voxel to ensure uniform distribution, 1.0 means there will only be 1 point for every 1ang voxel
            "distance": 1.05,  # This is the target value for the distance which sampled points will exist with respect to the set of smooth distance functions
            "variance": 0.1,  # Samples are kept with they are within this distance of the target distance
            "sup_sampling": 20,  # Number of points that are initially sampled around each atom according to a normal distribution centered at the atom's position and a std of 10ang
            "atom_feature_dim": 32,  # Dimension of the atom features extract using Graph Featurizer
            "emb_dims": 64,  # This is the final output learned embedding dimension of each surface point # CHANGE FROM 128 TO 64
            "in_channels": 42,  # Total number of geometric features + chemical features expected to feed to oritentation unit MLP
            "orientation_units": 42,  # Size of hidden layer in orientation unit MLP, kept the same as 'in_channels' in the original implementation
            "n_layers": 1,
            "radius": 9.0,  # this is the size of the convolutional surface patch being considered
            "dropout": 0.3,
            "device": device,
        }

        # Update with any additional arguments provided on initialization
        self.config.update(kwargs)

        # Initialize the protein surface encoder with the updated configuration
        self.protein_surface_encoder = dMaSIF(self.config)

    def forward(self, protein_surface_data):
        P = {}
        P["batch_atoms"] = protein_surface_data["atom_coords_batch"].to(
            self.config["device"]
        )
        P["atom_xyz"] = protein_surface_data["atom_coords"].to(self.config["device"])
        P["atom_features"] = protein_surface_data["atom_features"].to(
            self.config["device"]
        )
        P["xyz"] = protein_surface_data["surface_coords"].to(self.config["device"])
        P["normals"] = protein_surface_data["surface_normals"].to(self.config["device"])
        P["batch"] = protein_surface_data["surface_batch_idx"].to(self.config["device"])
        p_sur = iterate(self.protein_surface_encoder, P)
        return p_sur
