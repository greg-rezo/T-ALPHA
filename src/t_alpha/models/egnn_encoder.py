from torch import nn

from t_alpha.models.egnn import EGNN


class EGNNEncoder(nn.Module):
    def __init__(
        self,
        device: str,
        in_node_nf: int = 27,
        hidden_nf: int = 64,
        out_node_nf: int = 64,
        in_edge_nf: int = 6,
        act_fn: nn.Module = nn.SiLU(),
        n_layers: int = 12,
        residual: bool = True,
        attention: bool = True,
        normalize: bool = False,
        tanh: bool = False,
    ):
        super(EGNNEncoder, self).__init__()

        # Initialize the EGNN model
        self.egnn = EGNN(
            device=device,
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=out_node_nf,
            in_edge_nf=in_edge_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            residual=residual,
            attention=attention,
            normalize=normalize,
            tanh=tanh,
        )

    def forward(self, h, x, edges, edge_attr):
        # Pass through EGNN layers to get updated node features and coordinates
        h, x = self.egnn(h, x, edges, edge_attr)

        return h
