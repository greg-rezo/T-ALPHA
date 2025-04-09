import torch
from torch import nn
from torch.nn import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from t_alpha.models.egnn_encoder import EGNNEncoder
from t_alpha.models.surface_convnet import SurfaceConvNet
from t_alpha.models.utils import AttentionPooling, masked_mean_pool


class MetaModel(nn.Module):
    """
    MetaModel is a comprehensive neural network architecture designed for processing and integrating multiple
    biological data modalities, including protein surfaces, protein graphs, protein sequences, ligand properties,
    ligand graphs, ligand sequences, and complex graphs. The model utilizes transformer-based encoders and decoders
    to capture intricate relationships within and between these modalities.

    **Key Features:**
    - **Protein Components:**
        - *Surface*: Processes protein surface data using a convolutional network.
        - *Graph*: Encodes protein structural graphs using an EGNN encoder.
        - *Sequence*: Encodes protein sequences using transformer encoders/decoders.
    - **Ligand Components:**
        - *Properties*: Encodes ligand properties.
        - *Graph*: Encodes ligand structural graphs using an EGNN encoder.
        - *Sequence*: Encodes ligand sequences using transformer encoders/decoders.
    - **Complex Components:**
        - *Graph*: Encodes complex structural graphs using an EGNN encoder.
    - **Meta Transformer:**
        - Integrates processed protein, ligand, and complex features using transformer encoders/decoders.
    - **Attention Pooling and Output MLP:**
        - Applies attention-based pooling to aggregate features and produces the final output through a multi-layer perceptron.

    **Parameters:**
        Args:
            device: Device to run the model on ('gpu' or 'cpu'). Default is 'gpu'.
            batch_norm: If True, applies Batch Normalization layers. Default is True.
            use_protein_graph: If True, includes the protein graph component. Default is True.
            use_protein_surface: If True, includes the protein surface component. Default is True.
            use_protein_sequence: If True, includes the protein sequence component. Default is True.
            use_ligand_properties: If True, includes the ligand properties component. Default is True.
            use_ligand_graph: If True, includes the ligand graph component. Default is True.
            use_ligand_sequence: If True, includes the ligand sequence component. Default is True.
            use_complex_graph: If True, includes the complex graph component. Default is True.
    """

    def __init__(
        self,
        device: str,
        batch_norm: bool = True,
        use_protein_graph: bool = True,
        use_protein_surface: bool = True,
        use_protein_sequence: bool = True,
        use_ligand_properties: bool = True,
        use_ligand_graph: bool = True,
        use_ligand_sequence: bool = True,
        use_complex_graph: bool = True,
    ):
        super(MetaModel, self).__init__()
        self.batch_norm = batch_norm
        self.device = device
        self.use_protein_graph = use_protein_graph
        self.use_protein_surface = use_protein_surface
        self.use_protein_sequence = use_protein_sequence
        self.use_ligand_properties = use_ligand_properties
        self.use_ligand_graph = use_ligand_graph
        self.use_ligand_sequence = use_ligand_sequence
        self.use_complex_graph = use_complex_graph

        # PROTEIN #

        if self.use_protein_surface:
            # protein surface model
            self.protein_surface_model = SurfaceConvNet(device=device)

            # protein surface encoder
            self.protein_surface_encoder_layer = TransformerEncoderLayer(
                d_model=64, nhead=2, dim_feedforward=64, dropout=0.3, batch_first=True
            )
            self.protein_surface_encoder = TransformerEncoder(
                self.protein_surface_encoder_layer, num_layers=1
            )

            if self.use_protein_graph:
                # protein graph decoder
                self.protein_graph_decoder_layer = TransformerDecoderLayer(
                    d_model=64,
                    nhead=2,
                    dim_feedforward=64,
                    dropout=0.3,
                    batch_first=True,
                )
                self.protein_graph_decoder = TransformerDecoder(
                    self.protein_graph_decoder_layer, num_layers=1
                )

            if self.use_protein_sequence:
                # protein sequence decoder
                self.protein_sequence_decoder_layer = TransformerDecoderLayer(
                    d_model=64,
                    nhead=2,
                    dim_feedforward=64,
                    dropout=0.3,
                    batch_first=True,
                )
                self.protein_sequence_decoder = TransformerDecoder(
                    self.protein_sequence_decoder_layer, num_layers=1
                )

        else:
            # protein graph encoder
            self.protein_graph_encoder_layer = TransformerEncoderLayer(
                d_model=64, nhead=2, dim_feedforward=64, dropout=0.3, batch_first=True
            )
            self.protein_graph_encoder = TransformerEncoder(
                self.protein_graph_encoder_layer, num_layers=1
            )

            # protein sequence encoder
            self.protein_sequence_encoder_layer = TransformerEncoderLayer(
                d_model=64, nhead=2, dim_feedforward=64, dropout=0.3, batch_first=True
            )
            self.protein_sequence_encoder = TransformerEncoder(
                self.protein_sequence_encoder_layer, num_layers=1
            )

        if self.use_protein_graph:
            # protein graph model
            self.protein_graph_model = EGNNEncoder(
                device=device, in_node_nf=31, n_layers=4
            )
            # protein graph transformer output embedding layer
            self.protein_graph_transformer_output_embedding_layer = nn.Linear(1, 512)

        if self.use_protein_sequence:
            # protein sequence projector
            self.protein_sequence_projector = (
                nn.Sequential(nn.Linear(2560, 512), nn.BatchNorm1d(512), nn.ReLU())
                if batch_norm
                else nn.Sequential(nn.Linear(2560, 512), nn.ReLU())
            )
            # protein sequence embedding layer
            self.protein_sequence_embedding_layer = nn.Linear(1, 64)

        # combined protein output
        num_protein_components = 0
        if self.use_protein_surface:
            num_protein_components += 1
        if self.use_protein_graph:
            num_protein_components += 1
        if self.use_protein_sequence:
            num_protein_components += 1

        self.combined_pooled_protein_transformer_output_projector = nn.Linear(
            num_protein_components * 512, 512
        )

        # LIGAND #

        if self.use_ligand_properties:
            # ligand properties model
            self.ligand_properties_embedding_layer = nn.Linear(1, 64)

            # ligand properties encoder
            self.ligand_properties_encoder_layer = TransformerEncoderLayer(
                d_model=64, nhead=2, dim_feedforward=64, dropout=0.3, batch_first=True
            )
            self.ligand_properties_encoder = TransformerEncoder(
                self.ligand_properties_encoder_layer, num_layers=1
            )

            if self.use_ligand_graph:
                # ligand graph decoder
                self.ligand_graph_decoder_layer = TransformerDecoderLayer(
                    d_model=64,
                    nhead=2,
                    dim_feedforward=64,
                    dropout=0.3,
                    batch_first=True,
                )
                self.ligand_graph_decoder = TransformerDecoder(
                    self.ligand_graph_decoder_layer, num_layers=1
                )

            if self.use_ligand_sequence:
                # ligand sequence decoder
                self.ligand_sequence_decoder_layer = TransformerDecoderLayer(
                    d_model=64,
                    nhead=2,
                    dim_feedforward=64,
                    dropout=0.3,
                    batch_first=True,
                )
                self.ligand_sequence_decoder = TransformerDecoder(
                    self.ligand_sequence_decoder_layer, num_layers=1
                )

        else:
            # ligand graph encoder
            self.ligand_graph_encoder_layer = TransformerEncoderLayer(
                d_model=64, nhead=2, dim_feedforward=64, dropout=0.3, batch_first=True
            )
            self.ligand_graph_encoder = TransformerEncoder(
                self.ligand_graph_encoder_layer, num_layers=1
            )

            # ligand sequence encoder
            self.ligand_sequence_encoder_layer = TransformerEncoderLayer(
                d_model=64, nhead=2, dim_feedforward=64, dropout=0.3, batch_first=True
            )
            self.ligand_sequence_encoder = TransformerEncoder(
                self.ligand_sequence_encoder_layer, num_layers=1
            )

        if self.use_ligand_graph:
            # ligand graph model
            self.ligand_graph_model = EGNNEncoder(
                device=device, in_node_nf=27, n_layers=4
            )
            # ligand graph transformer output embedding layer
            self.ligand_graph_transformer_output_embedding_layer = nn.Linear(1, 512)

        if self.use_ligand_sequence:
            # ligand sequence projector
            self.ligand_sequence_projector = (
                nn.Sequential(nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU())
                if batch_norm
                else nn.Sequential(nn.Linear(768, 512), nn.ReLU())
            )
            # ligand sequence embedding layer
            self.ligand_sequence_embedding_layer = nn.Linear(1, 64)

        # combined ligand output
        combined_ligand_dim = 0
        if self.use_ligand_properties:
            combined_ligand_dim += 209
        if self.use_ligand_graph:
            combined_ligand_dim += 512
        if self.use_ligand_sequence:
            combined_ligand_dim += 512

        self.combined_pooled_ligand_transformer_output_projector = nn.Linear(
            combined_ligand_dim, 512
        )

        # COMPLEX #

        if self.use_complex_graph:
            # complex graph model
            self.complex_graph_model = EGNNEncoder(
                device=device, in_node_nf=33, n_layers=8, in_edge_nf=7
            )

            # complex graph embedding layer
            self.complex_graph_embedding_layer = nn.Linear(1, 512)

        # META TRANSFORMER

        if self.use_complex_graph:
            # complex encoder
            self.complex_encoder_layer = TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=64, dropout=0.3, batch_first=True
            )
            self.complex_encoder = TransformerEncoder(
                self.complex_encoder_layer, num_layers=2
            )

            # protein decoder
            self.protein_decoder_layer = TransformerDecoderLayer(
                d_model=64, nhead=4, dim_feedforward=64, dropout=0.3, batch_first=True
            )
            self.protein_decoder = TransformerDecoder(
                self.protein_decoder_layer, num_layers=2
            )

            # ligand decoder
            self.ligand_decoder_layer = TransformerDecoderLayer(
                d_model=64, nhead=4, dim_feedforward=64, dropout=0.3, batch_first=True
            )
            self.ligand_decoder = TransformerDecoder(
                self.ligand_decoder_layer, num_layers=2
            )

            # attention pooling
            self.meta_transformer_output_pooling = AttentionPooling(512 * 3, 512 * 3)

            # output MLP
            self.output_mlp = (
                nn.Sequential(
                    nn.Linear(512 * 3, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Linear(512, 1),
                )
                if batch_norm
                else nn.Sequential(
                    nn.Linear(512 * 3, 512), nn.ReLU(), nn.Linear(512, 1)
                )
            )

        else:
            # protein encoder
            self.protein_encoder_layer = TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=64, dropout=0.3, batch_first=True
            )
            self.protein_encoder = TransformerEncoder(
                self.protein_encoder_layer, num_layers=2
            )

            # ligand encoder
            self.ligand_encoder_layer = TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=64, dropout=0.3, batch_first=True
            )
            self.ligand_encoder = TransformerEncoder(
                self.ligand_encoder_layer, num_layers=2
            )

            # attention pooling
            self.meta_transformer_output_pooling = AttentionPooling(512 * 2, 512 * 2)

            # output MLP
            self.output_mlp = (
                nn.Sequential(
                    nn.Linear(512 * 2, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Linear(512, 1),
                )
                if batch_norm
                else nn.Sequential(
                    nn.Linear(512 * 2, 512), nn.ReLU(), nn.Linear(512, 1)
                )
            )

    def forward(self, data):
        # Move all tensor inputs to the device
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device)

        # PROTEIN #

        # protein surface
        if self.use_protein_surface:
            protein_surface_output = self.protein_surface_model(data)
            protein_surface_transformer_output = self.protein_surface_encoder(
                protein_surface_output
            )

        # protein graph
        if self.use_protein_graph:
            # protein graph output
            protein_graph_output = self.protein_graph_model(
                data["protein_graph"].node_feats,
                data["protein_graph"].node_coords,
                data["protein_graph"].edge_index,
                data["protein_graph"].edge_attr,
            )
            padded_protein_graph_output, protein_graph_mask = to_dense_batch(
                protein_graph_output, batch=data["protein_graph_batch"]
            )

            if self.use_protein_surface:
                # protein graph decoder output
                protein_graph_transformer_output = self.protein_graph_decoder(
                    padded_protein_graph_output,
                    protein_surface_transformer_output,
                    tgt_key_padding_mask=~protein_graph_mask,
                )

            else:
                # protein graph encoder output
                protein_graph_transformer_output = self.protein_graph_encoder(
                    padded_protein_graph_output,
                    src_key_padding_mask=~protein_graph_mask,
                )

            # pooled protein graph transformer output
            pooled_protein_graph_transformer_output = (
                masked_mean_pool(protein_graph_transformer_output, protein_graph_mask)
                .unsqueeze(1)
                .permute(0, 2, 1)
            )
            pooled_protein_graph_transformer_output = (
                self.protein_graph_transformer_output_embedding_layer(
                    pooled_protein_graph_transformer_output
                )
            )
            pooled_protein_graph_transformer_output = (
                pooled_protein_graph_transformer_output.permute(0, 2, 1)
            )

        # protein sequence
        if self.use_protein_sequence:
            # protein sequence output
            protein_sequence_output = self.protein_sequence_projector(
                data["esm_vector"]
            ).unsqueeze(2)
            protein_sequence_output_embedding = self.protein_sequence_embedding_layer(
                protein_sequence_output
            )

            if self.use_protein_surface:
                # protein sequence decoder output
                protein_sequence_transformer_output = self.protein_sequence_decoder(
                    protein_sequence_output_embedding,
                    protein_surface_transformer_output,
                )

            else:
                # protein sequence encoder output
                protein_sequence_transformer_output = self.protein_sequence_encoder(
                    protein_sequence_output_embedding
                )

        # combine the three protein transformer outputs
        if (
            self.use_protein_surface
            and self.use_protein_graph
            and self.use_protein_sequence
        ):
            combined_pooled_protein_transformer_output = torch.cat(
                [
                    protein_surface_transformer_output,
                    pooled_protein_graph_transformer_output,
                    protein_sequence_transformer_output,
                ],
                dim=1,
            ).permute(0, 2, 1)

        elif not self.use_protein_surface:
            combined_pooled_protein_transformer_output = torch.cat(
                [
                    pooled_protein_graph_transformer_output,
                    protein_sequence_transformer_output,
                ],
                dim=1,
            ).permute(0, 2, 1)

        elif not self.use_protein_graph:
            combined_pooled_protein_transformer_output = torch.cat(
                [
                    protein_surface_transformer_output,
                    protein_sequence_transformer_output,
                ],
                dim=1,
            ).permute(0, 2, 1)

        elif not self.use_protein_sequence:
            combined_pooled_protein_transformer_output = torch.cat(
                [
                    protein_surface_transformer_output,
                    pooled_protein_graph_transformer_output,
                ],
                dim=1,
            ).permute(0, 2, 1)

        final_protein_transformer_output = (
            self.combined_pooled_protein_transformer_output_projector(
                combined_pooled_protein_transformer_output
            ).permute(0, 2, 1)
        )

        # LIGAND #

        # ligand properties
        if self.use_ligand_properties:
            ligand_properties_output = self.ligand_properties_embedding_layer(
                data["rdkit_vector"].unsqueeze(2)
            )
            ligand_properties_transformer_output = self.ligand_properties_encoder(
                ligand_properties_output
            )

        # ligand sequence
        if self.use_ligand_sequence:
            # ligand sequence output
            ligand_sequence_output = self.ligand_sequence_projector(
                data["roberta_vector"]
            ).unsqueeze(
                2
            )  # convert from (batch, 768) to (batch, 512) to (batch, 512, 1)
            ligand_sequence_output_embedding = self.ligand_sequence_embedding_layer(
                ligand_sequence_output
            )  # convert from (batch, 512, 1) to (batch, 512, 128)

            if self.use_ligand_properties:
                # ligand sequence decoder output
                ligand_sequence_transformer_output = self.ligand_sequence_decoder(
                    ligand_sequence_output_embedding,
                    ligand_properties_transformer_output,
                )

            else:
                # ligand sequence encoder output
                ligand_sequence_transformer_output = self.ligand_sequence_encoder(
                    ligand_sequence_output_embedding
                )

        # ligand graph
        if self.use_ligand_graph:
            # ligand graph output
            ligand_graph_output = self.ligand_graph_model(
                data["ligand_graph"].node_feats,
                data["ligand_graph"].node_coords,
                data["ligand_graph"].edge_index,
                data["ligand_graph"].edge_attr,
            )
            padded_ligand_graph_output, ligand_graph_mask = to_dense_batch(
                ligand_graph_output, batch=data["ligand_graph_batch"]
            )

            if self.use_ligand_properties:
                # ligand graph decoder output
                ligand_graph_transformer_output = self.ligand_graph_decoder(
                    padded_ligand_graph_output,
                    ligand_sequence_transformer_output,
                    tgt_key_padding_mask=~ligand_graph_mask,
                )

            else:
                # ligand graph encoder output
                ligand_graph_transformer_output = self.ligand_graph_encoder(
                    padded_ligand_graph_output, src_key_padding_mask=~ligand_graph_mask
                )

            # pooled ligand graph transformer output
            pooled_ligand_graph_transformer_output = (
                masked_mean_pool(ligand_graph_transformer_output, ligand_graph_mask)
                .unsqueeze(1)
                .permute(0, 2, 1)
            )
            pooled_ligand_graph_transformer_output = (
                self.ligand_graph_transformer_output_embedding_layer(
                    pooled_ligand_graph_transformer_output
                )
            )
            pooled_ligand_graph_transformer_output = (
                pooled_ligand_graph_transformer_output.permute(0, 2, 1)
            )

        # combine the three ligand transformer outputs
        if (
            self.use_ligand_properties
            and self.use_ligand_graph
            and self.use_ligand_sequence
        ):
            combined_pooled_ligand_transformer_output = torch.cat(
                [
                    ligand_properties_transformer_output,
                    pooled_ligand_graph_transformer_output,
                    ligand_sequence_transformer_output,
                ],
                dim=1,
            ).permute(0, 2, 1)

        elif not self.use_ligand_properties:
            combined_pooled_ligand_transformer_output = torch.cat(
                [
                    pooled_ligand_graph_transformer_output,
                    ligand_sequence_transformer_output,
                ],
                dim=1,
            ).permute(0, 2, 1)

        elif not self.use_ligand_graph:
            combined_pooled_ligand_transformer_output = torch.cat(
                [
                    ligand_properties_transformer_output,
                    ligand_sequence_transformer_output,
                ],
                dim=1,
            ).permute(0, 2, 1)

        elif not self.use_ligand_sequence:
            combined_pooled_ligand_transformer_output = torch.cat(
                [
                    ligand_properties_transformer_output,
                    pooled_ligand_graph_transformer_output,
                ],
                dim=1,
            ).permute(0, 2, 1)

        final_ligand_transformer_output = (
            self.combined_pooled_ligand_transformer_output_projector(
                combined_pooled_ligand_transformer_output
            ).permute(0, 2, 1)
        )

        # COMPLEX #

        # complex graph output
        complex_graph_output = self.complex_graph_model(
            data["complex_graph"].node_feats,
            data["complex_graph"].node_coords,
            data["complex_graph"].edge_index,
            data["complex_graph"].edge_attr,
        )

        # pooled complex graph output
        pooled_complex_graph_output = (
            global_mean_pool(complex_graph_output, batch=data["complex_graph_batch"])
            .unsqueeze(1)
            .permute(0, 2, 1)
        )
        pooled_complex_graph_output = self.complex_graph_embedding_layer(
            pooled_complex_graph_output
        )
        pooled_complex_graph_output = pooled_complex_graph_output.permute(0, 2, 1)

        # META TRANSFORMER #

        if self.use_complex_graph:
            # complex encoder
            complex_transformer_output = self.complex_encoder(
                pooled_complex_graph_output
            )

            # protein decoder
            protein_transformer_output = self.protein_decoder(
                final_protein_transformer_output, complex_transformer_output
            )

            # ligand decoder
            ligand_transformer_output = self.ligand_decoder(
                final_ligand_transformer_output, complex_transformer_output
            )

            # concatenate
            meta_transformer_output = torch.cat(
                [
                    complex_transformer_output,
                    protein_transformer_output,
                    ligand_transformer_output,
                ],
                dim=1,
            ).permute(0, 2, 1)

        else:
            # protein encoder
            protein_transformer_output = self.protein_encoder(
                final_protein_transformer_output
            )

            # ligand encoder
            ligand_transformer_output = self.ligand_encoder(
                final_ligand_transformer_output
            )

            # concatenate
            meta_transformer_output = torch.cat(
                [protein_transformer_output, ligand_transformer_output], dim=1
            ).permute(0, 2, 1)

        # attention pooling
        output = self.meta_transformer_output_pooling(meta_transformer_output)

        # output
        output = self.output_mlp(output)

        return output
