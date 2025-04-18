# The classes in this file are adapted from the E(n) EGNN software corresponding
# to this paper: <https://arxiv.org/abs/2102.09844v3>

import torch
from torch import nn

from t_alpha.utils.graph_utils import unsorted_segment_mean, unsorted_segment_sum


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer (E_GCL)

    This layer is designed to be equivariant to rotations, translations, and reflections (E(n) symmetry).
    It updates both node features (h) and coordinates (x), ensuring that the layer preserves E(n) equivariance.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
    ):
        """
        Initializes the E(n)-equivariant convolutional layer.

        :param input_nf: Number of input node features (e.g., atom types).
        :param output_nf: Number of output node features.
        :param hidden_nf: Number of hidden features used in MLPs (for edges and nodes).
        :param edges_in_d: Number of edge features (e.g., bond type).
        :param act_fn: Activation function. SiLU (Swish) is used here for non-linearity.
        :param residual: Boolean flag to use residual connections (skip connections).
        :param attention: Boolean flag to use attention mechanism on edge features.
        :param normalize: Whether to normalize the coordinate differences.
        :param coords_agg: How to aggregate coordinates (options: "sum" or "mean").
        :param tanh: If True, applies Tanh activation to output to bound coordinate updates for stability.
        """

        # The number of features used in edge MLP is the sum of features of two connected nodes + edge features.
        super(E_GCL, self).__init__()
        input_edge = (
            input_nf * 2
        )  # Combine the features of the source and target nodes for edge modeling
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = (
            coords_agg  # Coordinate aggregation method: either sum or mean
        )
        self.tanh = tanh
        self.epsilon = 1e-8  # Small epsilon to avoid division by zero in normalization
        edge_coords_nf = (
            1  # One feature for relative distance between nodes (used in edge MLP)
        )

        """
        The edge MLP is responsible for updating edge features by considering node features 
        from both ends of the edge, their relative distance (||x_i - x_j||^2), and any additional edge features.
        This ensures that edge updates are dependent on node connections and their relative positions.
        """

        # Edge MLP: Models interactions between nodes and updates the edge features.
        # We concatenate the features of source and target nodes, relative distances (radial), and edge attributes.
        self.edge_mlp = nn.Sequential(
            nn.Linear(
                input_edge + edge_coords_nf + edges_in_d, hidden_nf
            ),  # First layer for processing edge information
            act_fn,  # Non-linear activation (SiLU/Swish)
            nn.Linear(
                hidden_nf, hidden_nf
            ),  # Another layer to project into hidden space
            act_fn,
        )  # Final activation function

        """
        Node MLP is used to update node features after aggregating information from the edges.
        The aggregation combines the current node features with the messages passed from edges. 
        The number of input features is the sum of hidden edge features and original node features.
        """

        # Node MLP: Updates node features based on the aggregated edge information.
        self.node_mlp = nn.Sequential(
            nn.Linear(
                hidden_nf + input_nf, hidden_nf
            ),  # Combine hidden edge features with node features
            act_fn,  # Apply activation function (SiLU/Swish)
            nn.Linear(hidden_nf, output_nf),
        )  # Output layer for node features

        """
        The coordinate MLP is responsible for updating the coordinates (positions) of the nodes. 
        This uses edge features to influence the movement of node coordinates, ensuring that the coordinates 
        are updated in an equivariant manner (i.e., preserving rotation and translation invariance).
        """

        # A small network (MLP) to calculate coordinate updates.
        #
        # This MLP updates node coordinates based on the edge features passed to it.
        # The update process for coordinates ensures the model is equivariant to rotations and translations,
        # which is crucial for modeling 3D structures like molecular interactions.
        #
        # According to the paper, this is a crucial part of ensuring that node positions
        # evolve based on their relative distances to one another while maintaining equivariance.
        # The key idea is to use the information contained in the hidden edge features
        # (i.e., information about connections between nodes and their distances) to update the node positions.
        #
        # In the equation for the coordinate update, the core idea is:
        #   Δx_i = Σ (x_i - x_j) * φ_x(h_ij)
        # where:
        #   - (x_i - x_j) is the coordinate difference between nodes i and j.
        #   - h_ij are the edge features (transformed via the edge MLP).
        #   - φ_x is the function modeled by the coordinate MLP.
        # This weighted sum over neighboring nodes ensures the equivariance of the system.

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(
            layer.weight, gain=0.001
        )  # Xavier initialization to stabilize training

        # Coordinate MLP to calculate the effect of the edge features on the node positions.
        # This MLP applies two linear transformations and a non-linear activation function (SiLU).
        # It uses Tanh to bound the output if `self.tanh` is True, which helps maintain stability by
        # constraining the coordinate updates to reasonable ranges. If `self.tanh` is False,
        # `nn.Identity()` is applied, which leaves the coordinates unchanged.
        self.coord_mlp = nn.Sequential(
            nn.Linear(
                hidden_nf, hidden_nf
            ),  # Hidden layer to transform the edge features for the coordinate update.
            act_fn,  # Non-linear activation (SiLU/Swish) to introduce non-linearity into the transformation.
            layer,  # The final layer outputs a scalar used to update the coordinates.
            (
                nn.Tanh() if self.tanh else nn.Identity()
            ),  # Apply Tanh if needed to bound the update; otherwise, do nothing.
        )

        # Attention mechanism (optional): This allows the model to assign different importance to different edges.
        # It computes a learned weight for each edge, scaling its effect during message passing.
        # This is useful when some connections should influence nodes more than others.
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(
                    hidden_nf, 1
                ),  # Attention weight is computed using hidden edge features.
                nn.Sigmoid(),  # Sigmoid ensures that the attention weights are between 0 and 1.
            )

    def edge_model(self, source, target, radial, edge_attr):
        """
        This function computes the updated edge features based on the node features
        of the connected nodes (source and target), their relative distances (radial),
        and any additional edge attributes (edge_attr).

        The core idea is to capture interactions between connected nodes and encode
        this information as edge features.

        Equation:
        The concatenated input includes:
        - Source node features (h_i)
        - Target node features (h_j)
        - Squared distance between nodes (||x_i - x_j||^2)
        - Optional edge attributes (e.g., bond type)

        The result of this operation is processed through the edge MLP to produce the updated edge features.
        """

        # Concatenate the node features and the radial distance (||x_i - x_j||^2)
        # If edge_attr is provided, include it in the concatenation.
        if edge_attr is None:
            out = torch.cat(
                [source, target, radial], dim=1
            )  # No explicit edge features, just nodes and distance
        else:
            out = torch.cat(
                [source, target, radial, edge_attr], dim=1
            )  # Include edge attributes as well

        # Pass the concatenated features through the edge MLP
        out = self.edge_mlp(out)

        # If attention is enabled, apply the learned attention weights to modulate the edge features
        if self.attention:
            att_val = self.att_mlp(out)  # Compute attention weight for each edge
            out = out * att_val  # Scale the edge features by attention weights

        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        """
        This function computes the updated node features by aggregating information
        from the edges connected to each node.

        The aggregation combines:
        - Current node features (x)
        - Aggregated edge features (messages passed through edges)
        - Optionally, additional node attributes (node_attr)

        Equation:
        - Aggregation is performed over the edges:
        h_i^{new} = f(h_i, Σ_j φ_edge(m_ij))
        where:
        - φ_edge(m_ij) represents the edge message passed from node j to node i.
        - The node features are updated based on the aggregated edge information.
        """

        # Extract source and target nodes from edge_index
        row, col = edge_index

        # Aggregate edge features for each node (unsorted_segment_sum aggregates based on node connections)
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))

        # Concatenate current node features with the aggregated edge features and optional node attributes
        if node_attr is not None:
            agg = torch.cat(
                [x, agg, node_attr], dim=1
            )  # Include node attributes in the concatenation
        else:
            agg = torch.cat([x, agg], dim=1)  # Just node and aggregated edge features

        # Update node features using the node MLP
        out = self.node_mlp(agg)

        # If residual connections are enabled, add the input node features (skip connection)
        if self.residual:
            out = x + out  # Residual connection: Add input features to the output

        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        """
        This function updates the coordinates of the nodes by aggregating coordinate differences
        between connected nodes, weighted by the edge features.

        Equation:
        - The update rule for node positions is:
          Δx_i = Σ_j (x_i - x_j) * φ_coord(edge_feat_ij)
        where:
        - (x_i - x_j) is the coordinate difference between nodes i and j.
        - φ_coord is the coordinate MLP applied to the edge features (edge_feat_ij).
        - Aggregation is performed over all edges connected to node i.
        """

        # Extract source and target nodes from edge_index
        row, col = edge_index

        # Compute the transformation for each coordinate difference using the coordinate MLP
        trans = coord_diff * self.coord_mlp(edge_feat)

        # Aggregate the transformations to update the coordinates
        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coords_agg parameter" % self.coords_agg)

        # Update the node coordinates by adding the aggregated transformations
        coord = coord + agg

        return coord

    def coord2radial(self, edge_index, coord):
        """
        This function computes the coordinate differences and radial distances (squared)
        between connected nodes.

        Equation:
        - The radial distance between two nodes i and j is:
          radial_ij = ||x_i - x_j||^2
        - The coordinate difference is:
          coord_diff_ij = x_i - x_j
        These are key inputs for the edge and coordinate update mechanisms.
        """

        # Extract source and target nodes from edge_index
        row, col = edge_index

        # Compute the coordinate difference for each edge
        coord_diff = coord[row] - coord[col]

        # Compute the squared distance (radial) between connected nodes
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        # If normalization is enabled, normalize the coordinate difference to unit length
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon  # Avoid division by zero
            coord_diff = coord_diff / norm  # Normalize the coordinate differences

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        """
        Forward pass through the E(n) Equivariant Convolutional Layer.

        1. Compute radial distances and coordinate differences between connected nodes.
        2. Update edge features based on node features and distances (via edge_model).
        3. Update node coordinates based on edge features (via coord_model).
        4. Update node features based on aggregated edge information (via node_model).

        :param h: Node features.
        :param edge_index: Defines which nodes are connected by edges.
        :param coord: Node coordinates.
        :param edge_attr: Edge features (optional).
        :param node_attr: Node attributes (optional).
        :return: Updated node features, updated coordinates, and updated edge attributes.
        """

        # Extract source and target node indices
        row, col = edge_index

        # Compute radial distances and coordinate differences between connected nodes
        radial, coord_diff = self.coord2radial(edge_index, coord)

        # Update edge features based on node features and radial distances
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        # Update node coordinates based on the edge features
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)

        # Update node features based on the aggregated edge information
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        # Return the updated node features, coordinates, and edge attributes
        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(
        self,
        device: str,
        in_node_nf: int,
        hidden_nf: int,
        out_node_nf: int,
        in_edge_nf: int = 0,
        act_fn: nn.Module = nn.SiLU(),
        n_layers: int = 4,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
    ):
        """
        :param in_node_nf: Number of input node features (e.g., atom types in a molecule).
        :param hidden_nf: Number of hidden features (size of hidden layers in GCL).
        :param out_node_nf: Number of output node features (dimension of output node embeddings).
        :param in_edge_nf: Number of input edge features (e.g., bond types, distances).
        :param device: Device to run the model (CPU or GPU).
        :param act_fn: Non-linearity to use in the layers (default is SiLU).
        :param n_layers: Number of E_GCL layers in the EGNN.
        :param residual: Whether to use residual connections (skip connections) in each layer. (We recommend keeping this True).
        :param attention: Whether to use attention on edges (learns edge importance).

        :param normalize: Whether to normalize the coordinate differences before updates such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: If True, applies Tanh activation to the output to bound coordinate updates. (Improves stability but may
                    decrease accuracy. We didn't use it in our paper.)

        This class builds an EGNN by stacking multiple E_GCL (Equivariant Graph Convolutional Layers).
        The layers update both the node features (h) and node coordinates (x) while maintaining
        E(n) equivariance. Residual connections help stabilize learning by re-adding input features
        after each layer.
        """

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        # Embedding the input node features to the hidden feature size.
        # This transforms the input features (such as atom types) to a hidden size before passing them through the layers.
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)

        # Output layer to project the hidden node features back to the desired output size.
        # For example, this could be projecting the node embeddings to a classification or regression target.
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        # Add multiple E_GCL layers to build the network.
        # Each layer updates both the node features (h) and coordinates (x) while maintaining E(n)-equivariance.
        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                E_GCL(
                    self.hidden_nf,  # Input node features for the current layer
                    self.hidden_nf,  # Output node features for the current layer
                    self.hidden_nf,  # Hidden layer size used within the GCL
                    edges_in_d=in_edge_nf,  # Number of input edge features (e.g., bond lengths, bond types)
                    act_fn=act_fn,  # Activation function for the GCL layers (default SiLU)
                    residual=residual,  # Use residual connections (important for deep networks)
                    attention=attention,  # Whether to use attention on edge features
                    normalize=normalize,  # Normalize coordinate differences (||x_i - x_j||)
                    tanh=tanh,
                ),
            )  # Whether to apply Tanh for bounding coordinate updates

        # Move the model to the designated device (CPU or GPU)
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        """
        Forward pass through the EGNN.

        The forward pass performs the following:
        1. Embeds the input node features.
        2. Passes the node features and coordinates through each E_GCL layer.
        3. Outputs the final transformed node features and updated coordinates.

        :param h: Node features (e.g., atom types)
        :param x: Node coordinates (e.g., 3D positions of atoms in the protein-ligand complex)
        :param edges: Defines which nodes are connected by edges (i.e., the graph structure)
        :param edge_attr: Edge features (e.g., bond types, bond lengths)
        :return: Updated node features and updated node coordinates
        """

        # Step 1: Embed the input node features into the hidden feature size
        h = self.embedding_in(h)

        # DEBUGGING NOTES: it seems that the activation and coordinates explode
        # during graph convolutions on the ligand-complex network. Not sure why
        # this is the case
        acts = list()
        coords = list()
        dists = list()

        # Step 2: Pass through each GCL layer, updating the node features and coordinates.
        # Each layer updates both the features (h) and positions (x) by using the connectivity and edge features.
        for i in range(self.n_layers):
            acts.append(h)
            coords.append(x)
            distances = torch.cdist(x, x)
            mask = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
            dists.append(distances[mask])
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
            if h.isnan().sum() > 0 or x.isnan().sum() > 0:
                print(
                    "Average activations per layer",
                    [t.abs().mean().item() for t in acts],
                )
                print("Minimal distance per layer:", [d.min().item() for d in dists])
                # breakpoint()

        # Step 3: Project the hidden node features to the output size using the final embedding layer
        h = self.embedding_out(h)

        # Return the updated node features and coordinates
        return h, x
