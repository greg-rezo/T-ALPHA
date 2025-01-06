# The functions in this file are adapted from the dMaSIF software corresponding
# to this paper: <https://www.biorxiv.org/content/10.1101/2020.12.28.424589v1.full>

from pykeops.torch import LazyTensor
import torch
import torch.nn.functional as F


def calculate_smoothed_normals(point_coords, scales=[1.0], batch=None, normals=None):
    """Returns a smooth field of normals, possibly at different scales.

    points, normals, scale(s)  ->      normals
    (N, 3), (N,3),   (S,)   ->  (N, 3) or (N, S, 3)

    We use the provided normals to smooth the discrete vector
    field using Gaussian windows whose radii are given in the list of "scales".

    If more than one scale is provided, normal fields are computed in parallel
    and returned in a single 3D tensor.

    Args:
        vertices (Tensor): (N,3) coordinates of  3D points.
        scale (list of floats, optional): (S,) radii of the Gaussian smoothing windows. Defaults to [1.].
        batch (integer Tensor, optional): batch vector, as in PyTorch_geometric. Defaults to None.
        normals (Tensor, optional): (N,3) raw normals vectors on the vertices. Defaults to None.

    Returns:
        (Tensor): (N,3) or (N,S,3) point normals.
    """

    # different scales to pass the Gaussian window over
    # shape (S,)
    scales = torch.Tensor(scales).type_as(point_coords)

    # Normal of a vertex = average of all normals in a ball of size "scale":

    # point coordinates in shape (N, 1, 3)
    x_i = LazyTensor(point_coords[:, None, :])

    # point coordinate in shape (1, M, 3)
    y_j = LazyTensor(point_coords[None, :, :])

    # normal vectors in shape (1, M, 3)
    v_j = LazyTensor(normals[None, :, :])

    # different scales to pass the Gaussian window over
    # shape (1, 1, S)
    s = LazyTensor(scales[None, None, :])

    # compute the squared Euclidean distance between each pair of points (vertices) and centers
    # shape (N, M, 1)
    D_ij = ((x_i - y_j) ** 2).sum(-1)

    # applies the Gaussian window function to these distances, resulting in a weight for each pair based on their distance and the current scale
    # shape (N, M, S)
    K_ij = (-D_ij / (2 * s**2)).exp()

    # K_ij.ranges is a tuple of data/batch identifying ranges and slices
    # Allows effective batch operations
    K_ij.ranges = diagonal_ranges(batch)

    # This product multiplies each element in pairwise weight matrix K_ij with the original normal vector
    U = K_ij.tensorprod(v_j)

    # This completes the Gaussian weighted sum of all normals to create the direction for the smoothed new vector
    U = U.sum(dim=1)
    U = U.view(-1, len(scales), 3)  # (N, S, 3)

    # Normalize the vector to create true normals
    normals = F.normalize(U, p=2, dim=-1)  # (N, 3) or (N, S, 3)
    return normals


def tangent_vectors(normals):
    """Returns a pair of vector fields u and v to complete the orthonormal basis [n,u,v].

    normals    ->  uv
    (N, S, 3)  ->  (N, S, 2, 3)

    This routine assumes that the 3D "normal" vectors are normalized.
    It is based on the 2017 paper from Pixar, "Building an orthonormal basis, revisited".

    Args:
        normals (Tensor): (N,S,3) normals `n_i`, i.e. unit-norm 3D vectors.

    Returns:
        (Tensor): (N,S,2,3) unit vectors `u_i` and `v_i` to complete
            the tangent coordinate systems `[n_i,u_i,v_i].
    """

    x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
    s = (2 * (z >= 0)) - 1.0  # = z.sign(), but =1. if z=0.
    a = -1 / (s + z)
    b = x * y * a

    uv = torch.stack((1 + s * x * x * a, s * b, -s * x, b, s + y * y * a, -y), dim=-1)
    uv = uv.view(uv.shape[:-1] + (2, 3))

    return uv


def curvatures(points, scales=[1.0], batch=None, normals=None, reg=0.01):
    """Returns a collection of mean (H) and Gauss (K) curvatures at different scales.

    points, faces, scales  ->  (H_1, K_1, ..., H_S, K_S)
    (N, 3), (3, N), (S,)   ->         (N, S*2)

    We rely on a very simple linear regression method, for all points:

      1. Estimate normals.
      2. Compute a local tangent frame.
      3. In a pseudo-geodesic Gaussian neighborhood at scale s,
         compute the two (2, 2) covariance matrices PPt and PQt
         between the displacement vectors "P = x_i - x_j" and
         the normals "Q = n_i - n_j", projected on the local tangent plane.
      4. Up to the sign, the shape operator S at scale s is then approximated
         as  "S = (reg**2 * I_2 + PPt)^-1 @ PQt".
      5. The mean and Gauss curvatures are the trace and determinant of
         this (2, 2) matrix.

    Args:
        points (Tensor): (N,3) coordinates of the points.
        scales (list of floats, optional): list of (S,) smoothing scales. Defaults to [1.].
        batch (integer Tensor, optional): batch vector, as in PyTorch_geometric. Defaults to None.
        normals (Tensor, optional): (N,3) field of "raw" unit normals. Defaults to None.
        reg (float, optional): small amount of Tikhonov/ridge regularization
            in the estimation of the shape operator. Defaults to .01.

    Returns:
        (Tensor): (N, S*2) tensor of mean and Gauss curvatures computed for
            every point at the required scales.
    """
    # Number of points, number of scales
    # Here, vertices are the protein surface xyz coordinates
    N, S = points.shape[0], len(scales)

    # These are ranges and slices of the batch structure
    # The ranges are tensors of shape [num_batches, 2]  with each row being [start_idx, end_idx] thats specifies the start and end indices of each batch's data in a concatenated structure
    # slices is a 1-indexed 1D tensors for retreiving the batch id for a given datapoint
    # Allows for effective batch processing
    ranges = diagonal_ranges(batch)

    # Compute the normals at different scales:
    # Shape (N, S, 3)
    normals_s = calculate_smoothed_normals(
        points, normals=normals, scales=scales, batch=batch
    )

    # Compute local tangent bases to the original normals,
    # so that for each point at each scale,
    # there is an orthonormal basis to define a local coordinate system
    # shape (N, S, 2, 3)
    uv_s = tangent_vectors(normals_s)
    features = []

    for s, scale in enumerate(scales):
        # Extract the relevant descriptors at the current scale:
        normals = normals_s[:, s, :].contiguous()  #  (N, 3)
        uv = uv_s[:, s, :, :].contiguous()  # (N, 2, 3)

        # Encode as symbolic tensors:
        # Points:
        x_i = LazyTensor(points.view(N, 1, 3))
        x_j = LazyTensor(points.view(1, N, 3))

        # Normals:
        n_i = LazyTensor(normals.view(N, 1, 3))
        n_j = LazyTensor(normals.view(1, N, 3))

        # Tangent bases:
        uv_i = LazyTensor(uv.view(N, 1, 6))

        # Calculation of the pseudo-geodesic squared distance
        # Multiplies the squared euclidean distance between all point pairs
        # with their corresponding inner product
        # shape (N, N, 1)
        d2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)

        # Apply gaussian window
        # shape (N, N, 1)
        window_ij = (-d2_ij / (2 * (scale**2))).exp()

        # Project on the tangent plane
        # shapes (N, N, 2)
        P_ij = uv_i.matvecmult(x_j - x_i)
        Q_ij = uv_i.matvecmult(n_j - n_i)

        # Concatenate for shape (N, N, 2+2)
        PQ_ij = P_ij.concat(Q_ij)

        # Covariances, with a scale-dependent weight:
        PPt_PQt_ij = P_ij.tensorprod(PQ_ij)  # (N, N, 2*(2+2))
        PPt_PQt_ij = window_ij * PPt_PQt_ij  #  (N, N, 2*(2+2))

        # Reduction - with batch support:
        PPt_PQt_ij.ranges = ranges
        PPt_PQt = PPt_PQt_ij.sum(1)  # (N, 2*(2+2))

        # Reshape to get the two covariance matrices:
        PPt_PQt = PPt_PQt.view(N, 2, 2, 2)
        PPt, PQt = PPt_PQt[:, :, 0, :], PPt_PQt[:, :, 1, :]  # (N, 2, 2), (N, 2, 2)

        # Add a small ridge regression:
        PPt[:, 0, 0] += reg
        PPt[:, 1, 1] += reg

        # (minus) Shape operator, i.e. the differential of the Gauss map:
        # = (PPt^-1 @ PQt) : simple estimation through linear regression
        # S = torch.solve(PQt, PPt).solution
        S = torch.linalg.solve(PPt, PQt)
        a, b, c, d = S[:, 0, 0], S[:, 0, 1], S[:, 1, 0], S[:, 1, 1]  # (N,)

        # Normalization
        mean_curvature = a + d
        gauss_curvature = a * d - b * c
        features += [mean_curvature.clamp(-1, 1), gauss_curvature.clamp(-1, 1)]

    features = torch.stack(features, dim=-1)
    return features


def diagonal_ranges(batch_x=None, batch_y=None):
    """
    Encodes the block-diagonal structure associated with batch vectors. This function calculates indices for diagonal blocks (or ranges) and slices for batch vectors.

    Parameters:
        batch_x (torch.Tensor): A tensor representing a batch vector that indicates the batch membership of each item.
        batch_y (torch.Tensor): An optional second tensor representing another batch vector similar to batch_x. If not provided, it is assumed to be the same as batch_x (symmetric case).

    Returns:
        tuple: A tuple containing the diagonal block ranges and slices for both batch vectors, or None if no batches are provided.
    """

    def ranges_slices(batch):
        """
        Helper function to calculate the ranges and slices indices for a given batch vector.

        Parameters:
            batch (torch.Tensor): Batch tensor where each element indicates its batch membership.

        Returns:
            tuple: Tuple containing two elements:
                   - ranges: A tensor of size [num_batches, 2] where each row contains the start and end indices for each batch.
                   - slices: A tensor of indices for each batch, useful for slicing operations.
        """

        # Count the number of elements in each batch
        Ns = batch.bincount()

        # Compute cumulative sum to get the end index for each batch
        indices = Ns.cumsum(0)

        # Append zero at the beginning and combine with indices
        ranges = torch.cat((0 * indices[:1], indices))

        # Create a 2D tensor of [start_idx, end_idx] for each batch
        ranges = (
            torch.stack((ranges[:-1], ranges[1:]))
            .t()
            .int()
            .contiguous()
            .to(batch.device)
        )

        # One-based indices for each batch
        slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

        return ranges, slices

    if batch_x is None and batch_y is None:
        return None  # Exit if no batch data is provided
    elif batch_y is None:
        batch_y = (
            batch_x  # Use batch_x for both if batch_y is not provided (symmetric case)
        )

    ranges_x, slices_x = ranges_slices(
        batch_x
    )  # Calculate ranges and slices for batch_x
    ranges_y, slices_y = ranges_slices(
        batch_y
    )  # Calculate ranges and slices for batch_y

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x
