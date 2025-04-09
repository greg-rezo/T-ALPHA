import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataListLoader

from t_alpha.training.losses import custom_loss


class MetaModelLightning(pl.LightningModule):
    """
    A custom PyTorch LightningModule for training and validating T-ALPHA.

    This module handles model training, validation, data processing, and optimization.
    It supports custom loss functions, metrics tracking (including Pearson correlation),
    and learning rate schedulers with warm-up and cosine annealing.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        batch_size (int): Batch size for training and validation.
        num_epochs (int): Total number of training epochs.
        batch_norm (bool): Flag to enable/disable batch normalization.
    """

    def __init__(
        self,
        model,
        train_dataset=None,
        val_dataset=None,
        batch_size=16,
        num_epochs=100,
        batch_norm=True,
        warmup_epochs=30,
        learning_rate=3e-4,
    ):
        super(MetaModelLightning, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.batch_norm = batch_norm
        self.warmup_epochs = warmup_epochs
        self.learning_rate = learning_rate

        # Assign datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Metrics tracking
        self.best_val_correlation = -1
        self.epochs_no_improve = 0

        # Loss function
        self.criterion = custom_loss

        # Automatic optimization is enabled by default in PyTorch Lightning
        self.automatic_optimization = True

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        loss, correlation, correlation_combined = self.process_batch(batch)

        # Log per-batch metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_correlation", correlation, on_step=True, on_epoch=True, prog_bar=True
        )

        # Log the combined correlation
        if not np.isnan(correlation_combined):
            self.log(
                "train_correlation_combined",
                correlation_combined,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        # Log learning rate
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        self.log(
            "learning_rate", current_lr, on_step=True, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, correlation, correlation_combined = self.process_batch(batch)

        # Log per-batch metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_correlation", correlation, on_step=False, on_epoch=True, prog_bar=True
        )

        # Log the combined correlation
        if not np.isnan(correlation_combined):
            self.log(
                "val_correlation_combined",
                correlation_combined,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return {"val_loss": loss, "val_correlation": correlation}

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )

        # Warm-up scheduler
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, total_iters=self.warmup_epochs
        )

        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=100 - self.warmup_epochs, eta_min=self.learning_rate * 0.1
        )

        # Combine schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )

        # Scheduler configuration for PyTorch Lightning
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [scheduler_config]

    def train_dataloader(self):
        return DataListLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return DataListLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def process_batch(self, batch):
        device = self.device  # Use the device managed by Lightning

        # Initialize lists for accumulating data

        # PROTEIN #
        atom_coords_list = []
        atom_features_list = []
        surface_coords_list = []
        surface_normals_list = []
        surface_batch_idx_list = []
        atom_coords_batch_list = []

        # Protein sequence data
        esm_vector_list = []

        # Protein graph data
        protein_graph_list = []
        protein_graph_batch_list = []

        # LIGAND #
        rdkit_vector_list = []
        roberta_vector_list = []
        ligand_graph_list = []
        ligand_graph_batch_list = []

        # COMPLEX #
        complex_graph_list = []
        complex_graph_batch_list = []

        # GLOBAL
        operator_list = []
        label_list = []
        pdbid_list = []

        # Before concatenating, reshape vectors if needed
        reshaped_esm_vector_list = []
        reshaped_rdkit_vector_list = []
        reshaped_roberta_vector_list = []

        for idx, data_item in enumerate(batch):
            # Ensure data is on the correct device
            data_item = data_item.to(device)

            # PROTEIN #
            atom_coords_list.append(data_item.atom_coords)
            atom_features_list.append(data_item.atom_features)
            surface_coords_list.append(data_item.surface_coords)
            surface_normals_list.append(data_item.surface_normals)
            surface_batch_idx_list.append(
                torch.full(
                    (data_item.surface_coords.size(0),),
                    idx,
                    dtype=torch.long,
                    device=device,
                )
            )
            atom_coords_batch_list.append(
                torch.full(
                    (data_item.atom_coords.size(0),),
                    idx,
                    dtype=torch.long,
                    device=device,
                )
            )

            # Protein sequence data
            esm_vector_list.append(data_item.esm_vector)

            # Protein graph data
            protein_graph_data = Data(
                node_feats=data_item.node_features,
                node_coords=data_item.protein_coords,
                edge_index=data_item.edge_index,
                edge_attr=data_item.edge_attr,
            )

            protein_graph_list.append(protein_graph_data)
            protein_graph_batch_list.append(
                torch.full(
                    (data_item.node_features.size(0),),
                    idx,
                    dtype=torch.long,
                    device=device,
                )
            )

            # LIGAND #
            rdkit_vector_list.append(data_item.rdkit_vector)
            roberta_vector_list.append(data_item.roberta_vector)
            ligand_graph_data = Data(
                node_feats=data_item.ligand_features,
                node_coords=data_item.ligand_coords,
                edge_index=data_item.ligand_edges,
                edge_attr=data_item.ligand_edge_attr,
            )
            ligand_graph_list.append(ligand_graph_data)
            ligand_graph_batch_list.append(
                torch.full(
                    (data_item.ligand_features.size(0),),
                    idx,
                    dtype=torch.long,
                    device=device,
                )
            )

            # COMPLEX #
            complex_graph_data = Data(
                node_feats=data_item.complex_features,
                node_coords=data_item.complex_coords,
                edge_index=data_item.complex_edges,
                edge_attr=data_item.complex_edge_attr,
            )
            complex_graph_list.append(complex_graph_data)
            complex_graph_batch_list.append(
                torch.full(
                    (data_item.complex_features.size(0),),
                    idx,
                    dtype=torch.long,
                    device=device,
                )
            )

            # GLOBAL
            operator_list.append(data_item.operator)

            label_list.append(data_item.label)

            pdbid_list.append(data_item.pdbid)

        for esm_vector in esm_vector_list:
            if esm_vector.dim() == 1:  # Check if the vector has only 1 dimension
                esm_vector = esm_vector.unsqueeze(
                    0
                )  # Add an extra dimension to make it 2D
            reshaped_esm_vector_list.append(esm_vector)

        # Now concatenate the reshaped vectors
        batch_esm_vector = torch.cat(reshaped_esm_vector_list, dim=0)

        for rdkit_vector in rdkit_vector_list:
            if rdkit_vector.dim() == 1:  # Check if the vector has only 1 dimension
                rdkit_vector = rdkit_vector.unsqueeze(
                    0
                )  # Add an extra dimension to make it 2D
            reshaped_rdkit_vector_list.append(rdkit_vector)

        batch_rdkit_vector = torch.cat(reshaped_rdkit_vector_list, dim=0)

        for roberta_vector in roberta_vector_list:
            if roberta_vector.dim() == 1:  # Check if the vector has only 1 dimension
                roberta_vector = roberta_vector.unsqueeze(
                    0
                )  # Add an extra dimension to make it 2D
            reshaped_roberta_vector_list.append(roberta_vector)

        batch_roberta_vector = torch.cat(reshaped_roberta_vector_list, dim=0)

        # Concatenate data

        # PROTEIN #
        batch_atom_coords_batch = torch.cat(atom_coords_batch_list, dim=0)
        batch_atom_coords = torch.cat(atom_coords_list, dim=0)
        batch_atom_features = torch.cat(atom_features_list, dim=0)
        batch_surface_coords = torch.cat(surface_coords_list, dim=0)
        batch_surface_normals = torch.cat(surface_normals_list, dim=0)
        batch_surface_batch_idx = torch.cat(surface_batch_idx_list, dim=0)

        # Protein graph data
        batch_protein_graph = Batch.from_data_list(protein_graph_list)

        batch_protein_batch = torch.cat(protein_graph_batch_list, dim=0)

        # LIGAND #
        batch_ligand_graph = Batch.from_data_list(ligand_graph_list).to(device)
        batch_ligand_batch = torch.cat(ligand_graph_batch_list, dim=0)

        # COMPLEX #
        batch_complex_graph = Batch.from_data_list(complex_graph_list).to(device)
        batch_complex_batch = torch.cat(complex_graph_batch_list, dim=0)

        # GLOBAL
        operator_list = [chr(op[0]) for op in operator_list]

        label_list = torch.tensor(
            label_list, dtype=torch.float32, device=device
        ).unsqueeze(1)

        data = {
            "esm_vector": batch_esm_vector.to(device),
            "rdkit_vector": batch_rdkit_vector.to(device),
            "roberta_vector": batch_roberta_vector.to(device),
            "atom_coords_batch": batch_atom_coords_batch.to(device),
            "atom_coords": batch_atom_coords.to(device),
            "atom_features": batch_atom_features.to(device),
            "surface_coords": batch_surface_coords.to(device),
            "surface_normals": batch_surface_normals.to(device),
            "surface_batch_idx": batch_surface_batch_idx.to(device),
            "protein_graph": batch_protein_graph,
            "protein_graph_batch": batch_protein_batch.to(device),
            "ligand_graph": batch_ligand_graph,
            "ligand_graph_batch": batch_ligand_batch.to(device),
            "complex_graph": batch_complex_graph,
            "complex_graph_batch": batch_complex_batch.to(device),
            "operator": operator_list,
            "label": label_list,
            "pdbid": pdbid_list,
        }

        # Forward pass
        output = self(data)

        target = data["label"]
        operator = data["operator"]

        # Compute loss
        loss = self.criterion(output, target, operator)

        # Compute correlation
        output_np = output.detach().cpu().numpy().flatten()
        target_np = target.detach().cpu().numpy().flatten()
        correlation = pearsonr(output_np, target_np)[0]

        # Convert operator list to numpy array
        operator_array = np.array(operator)

        # Compute combined correlation for operators '=' or '~'
        combined_mask = np.isin(operator_array, ["=", "~"])
        if np.any(combined_mask):
            output_combined = output_np[combined_mask]
            target_combined = target_np[combined_mask]
            correlation_combined = pearsonr(output_combined, target_combined)[0]
        else:
            correlation_combined = np.nan  # or any default value you prefer

        # Return loss, overall correlation, and the combined correlation
        return loss, correlation, correlation_combined
