from torch_geometric.nn import knn


def knn_atoms(x, y, x_batch, y_batch, k):
    # x: (N, D) - reference points (atoms)
    # y: (M, D) - query points (surface points)
    # x_batch: (N,) - batch indices for x
    # y_batch: (M,) - batch indices for y

    # Compute kNN using PyTorch Geometric's knn function
    edge_index = knn(x, y, k=k, batch_x=x_batch, batch_y=y_batch)

    # Extract indices and reshape
    idx = edge_index[0].view(-1, k)  # Indices in x (reference), shape: (M, k)

    # Compute distances for the k nearest neighbors
    # Unsqueeze y to match dimensions for broadcasting
    dists = ((y.unsqueeze(1) - x[idx]) ** 2).sum(dim=-1)  # Shape: (M, k)

    return idx, dists
