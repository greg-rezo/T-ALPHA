# The classes in this file are adapted from the dMaSIF software corresponding
# to this paper: <https://www.biorxiv.org/content/10.1101/2020.12.28.424589v1.full>

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykeops.torch import LazyTensor

from t_alpha.utils.geometry import diagonal_ranges, tangent_vectors


class dMaSIFConv(nn.Module):
    def __init__(
        self, device, in_channels=1, out_channels=1, radius=1.0, hidden_units=None
    ):
        """Creates the KeOps convolution layer.

        I = in_channels  is the dimension of the input features, which is the total number of geometric features + chemical features
        O = out_channels is the dimension of the output features for each point, this is the embedding dimensions from the config
        H = hidden_units is the dimension of the intermediate representation
        radius is the size of the pseudo-geodesic Gaussian window w_ij = W(d_ij)


        This affordable layer implements an elementary "convolution" operator
        on a cloud of N points (x_i) in dimension 3 that we decompose in three steps:

          1. Apply the MLP "net_in" on the input features "f_i". (N, I) -> (N, H)

          2. Compute H interaction terms in parallel with:
                  f_i = sum_j [ w_ij * conv(P_ij) * f_j ]
            In the equation above:
              - w_ij is a pseudo-geodesic window with a set radius.
              - P_ij is a vector of dimension 3, equal to "x_j-x_i"
                in the local oriented basis at x_i.
              - "conv" is an MLP from R^3 to R^H:
                 - with 2 linear layers and C=8 intermediate "cuts" otherwise.
              - "*" is coordinate-wise product.
              - f_j is the vector of transformed features.

          3. Apply the MLP "net_out" on the output features. (N, H) -> (N, O)


        A more general layer would have implemented conv(P_ij) as a full
        (H, H) matrix instead of a mere (H,) vector... At a much higher
        computational cost. The reasoning behind the code below is that
        a given time budget is better spent on using a larger architecture
        and more channels than on a very complex convolution operator.
        Interactions between channels happen at steps 1. and 3.,
        whereas the (costly) point-to-point interaction step 2.
        lets the network aggregate information in spatial neighborhoods.

        Args:
            in_channels (int, optional): numper of input features per point. Defaults to 1.
            out_channels (int, optional): number of output features per point. Defaults to 1.
            radius (float, optional): deviation of the Gaussian window on the
                quasi-geodesic distance `d_ij`. Defaults to 1..
            hidden_units (int, optional): number of hidden features per point.
                Defaults to out_channels.
        """

        super(dMaSIFConv, self).__init__()
        self.device = device

        self.Input = in_channels
        self.Output = out_channels
        self.Radius = radius
        self.Hidden = self.Output if hidden_units is None else hidden_units
        self.Cuts = 8  # Number of hidden units for the 3D MLP Filter.

        # For performance reasons, we cut our "hidden" vectors
        # in n_heads "independent heads" of dimension 8.
        self.heads_dim = 8  # 4 is probably too small; 16 is certainly too big

        # We accept "Hidden" dimensions of size 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, ...
        if self.Hidden < self.heads_dim:
            self.heads_dim = self.Hidden

        if self.Hidden % self.heads_dim != 0:
            raise ValueError(
                f"The dimension of the hidden units ({self.Hidden})"
                + f"should be a multiple of the heads dimension ({self.heads_dim})."
            )
        else:
            self.n_heads = self.Hidden // self.heads_dim

        # Transformation of the input features:
        self.net_in = nn.Sequential(
            nn.Linear(self.Input, self.Hidden),  # (H, I) + (H,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.Hidden, self.Hidden),  # (H, H) + (H,)
            nn.LeakyReLU(negative_slope=0.2),
        ).to(self.device)  #  (H,)
        self.norm_in = nn.GroupNorm(4, self.Hidden).to(self.device)

        self.conv = nn.Sequential(
            nn.Linear(3, self.Cuts),  # (C, 3) + (C,)
            nn.ReLU(),  # KeOps does not support well LeakyReLu
            nn.Linear(self.Cuts, self.Hidden),
        ).to(self.device)  # (H, C) + (H,)

        # Transformation of the output features:
        self.net_out = nn.Sequential(
            nn.Linear(self.Hidden, self.Output),  # (O, H) + (O,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.Output, self.Output),  # (O, O) + (O,)
            nn.LeakyReLU(negative_slope=0.2),
        ).to(self.device)  #  (O,)

        self.norm_out = nn.GroupNorm(4, self.Output).to(self.device)

        # Custom initialization for the MLP convolution filters:
        # we get interesting piecewise affine cuts on a normalized neighborhood.
        with torch.no_grad():
            nn.init.normal_(self.conv[0].weight).to(self.device)
            nn.init.uniform_(self.conv[0].bias).to(self.device)
            self.conv[0].bias *= 0.8 * (self.conv[0].weight ** 2).sum(-1).sqrt()

            nn.init.uniform_(
                self.conv[2].weight,
                a=-1 / np.sqrt(self.Cuts),
                b=1 / np.sqrt(self.Cuts),
            ).to(self.device)
            nn.init.normal_(self.conv[2].bias).to(self.device)
            self.conv[2].bias *= 0.5 * (self.conv[2].weight ** 2).sum(-1).sqrt()

    def forward(self, points, nuv, features, ranges=None):
        """Performs a quasi-geodesic interaction step.

        points, local basis, in features  ->  out features
        (N, 3),   (N, 3, 3),    (N, I)    ->    (N, O)

        This layer computes the interaction step of Eq. (7) in the paper,
        in-between the application of two MLP networks independently on all
        feature vectors.

        Args:
            points (Tensor): (N,3) point coordinates `x_i`.
            nuv (Tensor): (N,3,3) local coordinate systems `[n_i,u_i,v_i]`.
            features (Tensor): (N,I) input feature vectors `f_i`.
            ranges (6-uple of integer Tensors, optional): low-level format
                to support batch processing, as described in the KeOps documentation.
                In practice, this will be built by a higher-level object
                to encode the relevant "batch vectors" in a way that is convenient
                for the KeOps CUDA engine. Defaults to None.

        Returns:
            (Tensor): (N,O) output feature vectors `f'_i`.
        """

        # Learn a transformation of the input features
        # (N, geometric+chem feats) -> (N, H)
        features = self.net_in(features)

        # shape (1,H,N)
        features = features.transpose(1, 0)[None, :, :]
        features = self.norm_in(features)

        # shape (N, H)
        features = features[0].transpose(1, 0).contiguous()

        # Compute the local "shape contexts"

        # Normalize the kernel radius
        # Shape (N,3)
        points = points / (math.sqrt(2.0) * self.Radius)

        # Points
        x_i = LazyTensor(points[:, None, :])  # (N, 1, 3)
        x_j = LazyTensor(points[None, :, :])  # (1, N, 3)

        # shape (N,3)
        # add detach if needed here
        normals = nuv[:, 0, :].contiguous()

        # Local bases
        # shape (N, 1, 9)
        nuv_i = LazyTensor(nuv.view(-1, 1, 9))

        # Normals
        n_i = nuv_i[:3]  # (N, 1, 3)
        n_j = LazyTensor(normals[None, :, :])  # (1, N, 3)

        # To avoid register spilling when using large embeddings, we perform our KeOps reduction
        # over the vector of length "self.Hidden = self.n_heads * self.heads_dim"
        # as self.n_heads reduction over vectors of length self.heads_dim (= "Hd" in the comments).
        head_out_features = []

        # n_heads is hidden_dim/head_dim
        for head in range(self.n_heads):
            # Extract a slice of width Hd from the feature array
            # head_features becomes shape (num_of_points_in_batch, hidden_dimension)
            head_start = head * self.heads_dim
            head_end = head_start + self.heads_dim
            head_features = features[
                :, head_start:head_end
            ].contiguous()  # (N, H) -> (N, Hd)

            # Features:
            f_j = LazyTensor(head_features[None, :, :])  # (1, N, Hd)

            # Convolution parameters
            # shape is (cuts, 3), (cuts,)
            A_1, B_1 = self.conv[0].weight, self.conv[0].bias

            # Extract a slice of Hd lines: (H, C) -> (Hd, C)
            A_2 = self.conv[2].weight[head_start:head_end, :].contiguous()
            # Extract a slice of Hd coefficients: (H,) -> (Hd,)
            B_2 = self.conv[2].bias[head_start:head_end].contiguous()
            a_1 = LazyTensor(A_1.view(1, 1, -1))  # (1, 1, C*3)
            b_1 = LazyTensor(B_1.view(1, 1, -1))  # (1, 1, C)
            a_2 = LazyTensor(A_2.view(1, 1, -1))  # (1, 1, Hd*C)
            b_2 = LazyTensor(B_2.view(1, 1, -1))  # (1, 1, Hd)

            # 2.c Pseudo-geodesic window:
            # Pseudo-geodesic squared distance:
            d2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)  # (N, N, 1)
            # Gaussian window:
            window_ij = (-d2_ij).exp()  # (N, N, 1)

            # 2.d Local MLP:
            # Project to local coordinates:
            X_ij = nuv_i.matvecmult(
                x_j - x_i
            )  # (N, N, 9) "@" (N, N, 3) = (N, N, 3) (there is pykeops implicit broadcasting here)

            # Manually apply with efficient Pykeops MLP:
            X_ij = a_1.matvecmult(X_ij) + b_1  # (N, N, C)
            X_ij = X_ij.relu()  # (N, N, C)
            X_ij = a_2.matvecmult(X_ij) + b_2  # (N, N, Hd)
            X_ij = X_ij.relu()
            # 2.e Actual computation:
            F_ij = window_ij * X_ij * f_j  # (N, N, Hd)
            F_ij.ranges = ranges  # Support for batches and/or block-sparsity

            head_out_features.append(
                ContiguousBackward().apply(F_ij.sum(dim=1))
            )  # (N, Hd)

        # Concatenate the result of our n_heads "attention heads":
        features = torch.cat(head_out_features, dim=1)  # n_heads * (N, Hd) -> (N, H)

        # 3. Transform the output features: ------------------------------------
        features = self.net_out(features)  # (N, H) -> (N, O)
        features = features.transpose(1, 0)[None, :, :]  # (1,O,N)
        features = self.norm_out(features)
        features = features[0].transpose(1, 0).contiguous()
        return features


class ContiguousBackward(torch.autograd.Function):
    """
    Function to ensure contiguous gradient in backward pass. To be applied after PyKeOps reduction.
    N.B.: This workaround fixes a bug that will be fixed in ulterior KeOp releases.
    """

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()


class dMaSIFConv_seg(torch.nn.Module):
    def __init__(self, config, in_channels, out_channels, n_layers, radius=9.0):
        super(dMaSIFConv_seg, self).__init__()

        self.name = "dMaSIFConv_seg_keops"
        self.radius = radius
        self.I, self.O = in_channels, out_channels
        self.device = config["device"]

        # layers of the convolution, defaults to 1
        self.layers = nn.ModuleList(
            [dMaSIFConv(self.device, self.I, self.O, radius, self.O)]
            + [
                dMaSIFConv(self.device, self.O, self.O, radius, self.O)
                for i in range(n_layers - 1)
            ]
        ).to(self.device)

        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.O, self.O), nn.ReLU(), nn.Linear(self.O, self.O)
                )
                for i in range(n_layers)
            ]
        ).to(self.device)

        self.linear_transform = nn.ModuleList(
            [nn.Linear(self.I, self.O)]
            + [nn.Linear(self.O, self.O) for i in range(n_layers - 1)]
        ).to(self.device)

    def forward(self, features):
        # Load in xyz of points
        # Shape (total_num_points, 3)
        points = self.points

        # Load in locally oriented basis
        # Shape (total_num_points, 3, 3)
        nuv = self.nuv

        # Load in batching data
        ranges = self.ranges

        # Chemical + Geometric features for each point
        # Shape (N, Chemical + Geometric features)
        x = features

        for i, layer in enumerate(self.layers):
            # output shape is (points_in_batch, embedding_dim)
            x_i = layer(points, nuv, x, ranges)

            # output shape is (points_in_batch, embedding_dim)
            x_i = self.linear_layers[i](x_i)

            # learn another representatino of the chemical+geometric embeddings
            # shape is (points_in_batch, embedding_dim)
            x = self.linear_transform[i](x)

            # residual connection with the newly learned Chemical + Geometric feature embeddings
            # shape is (points_in_batch, embedding_dim)
            x = x + x_i

        # shape is (points_in_batch, embedding_dim)
        return x

    def load_locally_oriented_bases(self, xyz, normals=None, weights=None, batch=None):
        """
        Input arguments:
        - xyz, a point cloud encoded as an (N, 3) Tensor.
        - weights, importance weights for the orientation estimation, encoded as an (N, 1) Tensor.
        - radius, the scale used to estimate the local normals.
        - a batch vector, following PyTorch_Geometric's conventions.

        The routine updates the model attributes:
        - points, i.e. the point cloud itself,
        - nuv, a local oriented basis in R^3 for every point,
        - ranges, custom KeOps syntax to implement batch processing.
        """

        # Save information for later use in convolution
        self.points = xyz
        self.batch = batch
        self.normals = normals
        self.weights = weights

        # These are ranges and slices of the batch structure
        # The ranges are tensors of shape [num_batches, 2]  with each row being [start_idx, end_idx] thats specifies the start and end indices of each batch's data in a concatenated structure
        # slices is a 1-indexed 1D tensors for retreiving the batch id for a given datapoint
        # Allows for effective batch processing
        self.ranges = diagonal_ranges(batch)

        # Normalize the scale
        points = xyz / self.radius

        # Generate two orthogonal vector tangent to the normals
        # shape (N,2,3)
        tangent_bases = tangent_vectors(normals)

        # Steer the tangent bases according to the gradient of "weights"
        # Orientation scores:
        weights_j = LazyTensor(weights.view(1, -1, 1))  # (1, N, 1)

        # Points
        x_i = LazyTensor(points[:, None, :])  # (N, 1, 3)
        x_j = LazyTensor(points[None, :, :])  # (1, N, 3)

        # Normals
        n_i = LazyTensor(normals[:, None, :])  # (N, 1, 3)
        n_j = LazyTensor(normals[None, :, :])  # (1, N, 3)

        # Tangent basis
        # shape (N, 1, 6)
        uv_i = LazyTensor(tangent_bases.view(-1, 1, 6))

        # Calculation of the pseudo-geodesic squared distance
        # Multiplies the squared euclidean distance between all point pairs
        # with their corresponding inner product
        # shape (N, N, 1)
        rho2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)

        # Apply gaussian window to pseudo-geodesic squared distance
        # shape (N, N, 1)
        window_ij = (-rho2_ij).exp()

        # Project coordinates in the (u, v) basis (not oriented yet)
        # shape (N, N, 2)
        X_ij = uv_i.matvecmult(x_j - x_i)

        # Currently, the pair of tangent vectors are only defined in the tangent plane.
        # To solve this, the first tangent vector is oriented along the geometric
        # gradient of orientation scores.
        # This gradient is approximated using the derivative of a Guassian filter
        # on the tangent plane in the form of a quasi-geodesic convolution
        # This is outlined in Equation 6 of https://doi.org/10.1109/CVPR46437.2021.01502

        # Gaussian window of pseudo-geodesic squared distance
        # is multiplied by the learnable weights and applied to
        # the vectors tangent to the normals
        # (N, N, 2)
        orientation_vector_ij = (window_ij * weights_j) * X_ij

        # Support for batch processing
        orientation_vector_ij.ranges = self.ranges

        # Sum and reduce to shape (N, 2)
        orientation_vector_i = orientation_vector_ij.sum(dim=1)

        # Add a corrective term for numerical stability
        orientation_vector_i = orientation_vector_i + 1e-5

        # Normalize
        orientation_vector_i = F.normalize(orientation_vector_i, p=2, dim=-1)

        # Save new tangent bases
        # Each ex_i, ey_i shape (N,1)
        ex_i, ey_i = (
            orientation_vector_i[:, 0][:, None],
            orientation_vector_i[:, 1][:, None],
        )

        # Re-orient the (u,v) basis
        # Each shape (N, 3)
        u_i, v_i = tangent_bases[:, 0, :], tangent_bases[:, 1, :]  # (N, 3)

        # Re-orient the original tangent basis into the new basis
        # produced from learned orientation
        # Shape (N,6)
        tangent_bases = torch.cat(
            (ex_i * u_i + ey_i * v_i, -ey_i * u_i + ex_i * v_i), dim=1
        ).contiguous()

        # Store the local 3D frame as an attribute
        self.nuv = torch.cat(
            (normals.view(-1, 1, 3), tangent_bases.view(-1, 2, 3)), dim=1
        )
