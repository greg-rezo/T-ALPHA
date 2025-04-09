from torch_geometric.utils import to_dense_batch


def iterate(net, protein):

    # passes the processed protein data (coords,normals, features, etc) to dMASIF
    outputs = net(protein)
    surface_emb, _ = to_dense_batch(outputs["embedding"], outputs["batch"])

    return surface_emb
