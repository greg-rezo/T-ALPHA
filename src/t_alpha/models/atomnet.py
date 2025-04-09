# The class in this file is adapted from the dMaSIF software corresponding
# to this paper: <https://www.biorxiv.org/content/10.1101/2020.12.28.424589v1.full>

import torch
import torch.nn as nn

from t_alpha.utils.knn import knn_atoms


class AtomNet_MP(nn.Module):
    def __init__(self, config: dict):
        super(AtomNet_MP, self).__init__()
        self.args = config
        self.device = config["device"]
        self.D = config["atom_feature_dim"]
        self.k_nearest = 16
        self.n_layers = 3

        # learns a representation of the input atomic features
        self.transform_atom_features = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.D, self.D),
        ).to(self.device)

        # this MLP combines information of nearest atom neighbors (input is features of both atoms + their distance)
        self.atom_atom_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for _ in range(self.n_layers)
            ]
        ).to(self.device)

        self.atom_atom_norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        ).to(self.device)

        # combines information between a point's nearest neighbor atoms
        self.atom_embedding_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for _ in range(self.n_layers)
            ]
        ).to(self.device)

        self.atom_embedding_norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for _ in range(self.n_layers)]
        ).to(self.device)

        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def atom_atom_forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        atom_features: torch.Tensor,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> torch.Tensor:
        # atomic coordinates
        # x: atom coordinates (reference points)
        # y: atom coordinates (query points)
        x = x.to(self.device)
        y = y.to(self.device)
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # Find nearest neighbors among atoms
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k=self.k_nearest + 1)
        idx = idx[:, 1:]  # Exclude self
        dists = dists[:, 1:]

        # number of atoms in batch
        num_points = atom_features.shape[0]

        out = atom_features
        for i in range(self.n_layers):
            _, num_atomtypes = out.size()

            # 'features' extracts the embeddings for neighbors using flattened indices from 'idx'
            features = out[idx.reshape(-1), :]

            # Concatenate the distances to the features to include spatial information
            # Shape (num_nearest_atoms*atoms_in_batch,num_atom_type+1)
            features = torch.cat(
                [features, dists.reshape(-1, 1).to(self.device)], dim=1
            )

            # Reshape 'features' back to (num_points, k_nearest, num_atomtypes + 1) to align with the number of neighbors and feature dimensions
            features = features.view(num_points, self.k_nearest, num_atomtypes + 1)

            # Repeat the central atom's features 'k_nearest' times and concatenate with neighbor features
            # This step effectively combines the target atom's features with each of its neighbors' features
            # shape is (num_atoms_in_batch, nearest_neighbors, 1+2*num_atomic_features)
            features = torch.cat(
                [out[:, None, :].repeat(1, self.k_nearest, 1), features], dim=-1
            )

            # Apply a neural network layer (MLP) to these combined features to compute messages
            # MLP outputs shape is (num_atoms_in_batch, nearest_neighbors, num_atomic_features)
            messages = self.atom_atom_mlp[i](features)

            # Sum the messages across neighbors to aggregate information
            # Shape is (num_atoms_in_batch, num_atomic_features)
            messages = messages.sum(1)

            # Update the atom's features by adding the transformed messages
            out = out + self.relu(self.atom_atom_norm[i](messages))

        # Shape is (num_atoms_in_batch, num_atomic_features)
        return out

    def atom_embedding_forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_atomtypes: torch.Tensor,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> torch.Tensor:
        # x: surface point coordinates (query points)
        # y: atom coordinates (reference points)
        x = x.to(self.device)
        y = y.to(self.device)
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # Find nearest atoms for each surface point
        idx, dists = knn_atoms(y, x, y_batch, x_batch, k=self.k_nearest)

        # number of surface points in batch
        num_points = x.shape[0]

        # number of atomtypes
        num_dims = y_atomtypes.shape[-1]

        # Initialize point embeddings as ones, shaped to match number of surface points and dimensions of atom types
        point_emb = torch.ones_like(x[:, 0])[:, None].repeat(1, num_dims)

        # Iterate over each layer for embedding calculations
        for i in range(self.n_layers):
            # Retrieve atom type features for closest atoms using the indices from knn
            features = y_atomtypes[idx.reshape(-1), :]

            # Concatenate distances to these features to add spatial context
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)

            # Reshape features to align with the number of surface points and the k nearest atoms
            features = features.view(num_points, self.k_nearest, num_dims + 1)

            # Repeat the initial point embeddings 'k_nearest' times and concatenate with neighbor atom features
            # This step effectively combines the target point's features with each of its atomic neighbors' features
            # shape is (num_points_in_batch, nearest_neighbors, 1+2*num_atom_types)
            features = torch.cat(
                [point_emb[:, None, :].repeat(1, self.k_nearest, 1), features], dim=-1
            )

            # Apply a neural network layer (MLP) to these combined features to generate messages
            # shape is (num_points_in_batch, nearest_neighbors, num_atom_types)
            messages = self.atom_embedding_mlp[i](features)

            # Aggregate messages by summing them up across the k dimensions
            # shape is (num_points_in_batch, num_atom_types)
            messages = messages.sum(1)

            # Update point embeddings by adding transformed messages
            point_emb = point_emb + self.relu(self.atom_embedding_norm[i](messages))

        # Return the final embeddings for each surface point
        # shape is (num_points_in_batch, num_atom_types)
        return point_emb

    def forward(
        self,
        xyz: torch.Tensor,
        atom_xyz: torch.Tensor,
        atom_features: torch.Tensor,
        batch: torch.Tensor,
        atom_batch: torch.Tensor,
    ) -> torch.Tensor:
        # learn a representation for atoms
        # shape (all_atoms_in_batch, num_of_atomic_features)
        atomic_feature_embeddings = self.transform_atom_features(atom_features)

        # pass information between nearest neighbor atoms
        # shape (all_atoms_in_batch, num_of_atomic_features)
        atomic_feature_embeddings = self.atom_atom_forward(
            atom_xyz, atom_xyz, atomic_feature_embeddings, atom_batch, atom_batch
        )

        # pass information between nearest neighboring atoms for each surface points
        # to generate an embedding for each surface point
        # shape (num_surface_points, num_of_atomic_features)
        surface_embeddings = self.atom_embedding_forward(
            xyz, atom_xyz, atomic_feature_embeddings, batch, atom_batch
        )
        return surface_embeddings
