# The class in this file is adapted from the dMaSIF software corresponding
# to this paper: <https://www.biorxiv.org/content/10.1101/2020.12.28.424589v1.full>

import torch
import torch.nn as nn

from t_alpha.models.atomnet import AtomNet_MP
from t_alpha.models.dmasif_conv import dMaSIFConv_seg
from t_alpha.utils.geometry import curvatures


class dMaSIF(nn.Module):
    def __init__(self, config: dict):
        super(dMaSIF, self).__init__()
        # Additional geometric features: mean and Gauss curvatures computed at different scales.
        self.curvature_scales = config["curvature_scales"]
        self.device = config["device"]

        # See these descriptions in SurfaceConvNet
        I = config["in_channels"]
        O = config["orientation_units"]
        E = config["emb_dims"]

        # Computes chemical features
        self.atomnet = AtomNet_MP(config)
        self.dropout = nn.Dropout(config["dropout"])

        # Post-processing. This is the orientation of the local frame (helping to learn to be invariant):
        self.orientation_scores = nn.Sequential(
            nn.Linear(I, O),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(O, 1),
        ).to(self.device)

        # Segmentation network:
        self.conv = dMaSIFConv_seg(
            config,
            in_channels=I,
            out_channels=E,
            n_layers=config["n_layers"],
            radius=config["radius"],
        )

    def features(self, P: dict) -> torch.Tensor:
        """Estimates geometric and chemical features from a protein surface or a cloud of atoms."""

        # Estimate the curvatures using the estimated normals
        # Returns a tensor shape (num_points_in_batch, 2*num_scales)
        # This information is the [mean curvature, gaussian curve] concatenated with the number of scales
        P_curvatures = curvatures(
            P["xyz"],
            normals=P["normals"],
            scales=self.curvature_scales,
            batch=P["batch"],
        )

        # Compute chemical features on-the-fly
        # Considers the each surface point and nearest atoms and their types to generate a point embedding
        # shape is (num_points_in_batch, num_atom_types)
        chemfeats = self.atomnet(
            P["xyz"], P["atom_xyz"], P["atom_features"], P["batch"], P["batch_atoms"]
        )

        # Returns the full feature embedding for each surface point by concatenating chemical and geometric features
        # shape is (num_points_in_batch, num_atomic_features+2*num_scales)...2*num_scales since its the mean curvature and gaussian curve at each scale
        return torch.cat([P_curvatures, chemfeats], dim=1).contiguous()

    def forward(self, P: dict) -> dict:
        """Embeds all points of a protein in a high-dimensional vector space."""

        # Returns the full feature embedding for each surface point by concatenating chemical and geometric features
        # shape is (num_points_in_batch, num_atom_types+2*num_scales)
        features = self.dropout(self.features(P))
        P["input_features"] = features

        torch.cuda.synchronize(device=features.device)

        # prepares for the quasi-geodesic convolution
        self.conv.load_locally_oriented_bases(
            P["xyz"],
            normals=P["normals"],
            weights=self.orientation_scores(
                features
            ),  # This reduces the geometric+chemical features to a single score scalar for each point
            batch=P["batch"],
        )

        # surface point embeddings
        P["embedding"] = self.conv(features)

        return P
