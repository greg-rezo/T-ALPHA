import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataListLoader

from t_alpha.data.full_model_dataset import MetaModelDataset
from t_alpha.models.full_model import MetaModel
from t_alpha.training.lightning_module import MetaModelLightning


def enable_dropout(model: torch.nn.Module) -> None:
    """
    Enable dropout layers during inference by setting them to training mode.

    Args:
        model (torch.nn.Module): The model containing dropout layers.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def perform_mc_dropout(
    ckpt_path: str,
    test_set_path: str,
    batch_size: int = 8,
    batch_norm: bool = True,
    device: str = "cpu",
    num_mc_samples: int = 100,
):
    """
    Perform Monte Carlo Dropout.

    Args:
        ckpt_path: Path to the checkpoint file (.ckpt).
        test_set_path: Path to the test dataset.
        batch_size: Batch size for DataLoader. Defaults to 8.
        batch_norm: Whether to use batch normalization. Defaults to True.
        device: Device to perform inference on ('cuda' or 'cpu').
                                Defaults to 'cuda' if available.
        num_mc_samples: Number of Monte Carlo samples for uncertainty estimation.
                       Defaults to 100.

    Returns:
        A DataFrame containing pdbid, prediction, and uncertainty for each sample.
    """
    # Verify that ckpt_path is a file
    if not os.path.isfile(ckpt_path):
        raise ValueError(
            f"Invalid checkpoint path: {ckpt_path}. Please provide a valid .ckpt file."
        )

    checkpoint_file = ckpt_path

    # Load the test dataset
    test_dataset = MetaModelDataset(test_set_path, device)

    # Create DataLoader for the test dataset
    test_loader = DataListLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Initialize the model
    model = MetaModel(device=device)

    # Initialize the LightningModule
    lightning_model = MetaModelLightning(
        model=model,
        batch_size=batch_size,
        num_epochs=0,  # Not training
        batch_norm=batch_norm,
    )

    # Load the checkpoint
    lightning_model = MetaModelLightning.load_from_checkpoint(
        checkpoint_file,
        model=model,
        batch_size=batch_size,
        num_epochs=0,
        batch_norm=batch_norm,
    )

    # Move the model to the appropriate device
    lightning_model.to(device)
    lightning_model.eval()

    # Enable dropout layers for uncertainty estimation
    enable_dropout(lightning_model.model)

    predictions = []
    uncertainties = []
    pdbids = []

    print("Starting MC dropout...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
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
            pdbid_list_batch = []

            for idx_item, data_item in enumerate(batch):
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
                        idx_item,
                        dtype=torch.long,
                        device=device,
                    )
                )
                atom_coords_batch_list.append(
                    torch.full(
                        (data_item.atom_coords.size(0),),
                        idx_item,
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
                        idx_item,
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
                        idx_item,
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
                        idx_item,
                        dtype=torch.long,
                        device=device,
                    )
                )

                # GLOBAL
                operator_list.append(data_item.operator)
                pdbid_list_batch.append(data_item.pdbid)

            # Concatenate data

            # PROTEIN #
            batch_atom_coords_batch = torch.cat(atom_coords_batch_list, dim=0)
            batch_atom_coords = torch.cat(atom_coords_list, dim=0)
            batch_atom_features = torch.cat(atom_features_list, dim=0)
            batch_surface_coords = torch.cat(surface_coords_list, dim=0)
            batch_surface_normals = torch.cat(surface_normals_list, dim=0)
            batch_surface_batch_idx = torch.cat(surface_batch_idx_list, dim=0)

            # Protein sequence data
            batch_esm_vector = torch.cat(esm_vector_list, dim=0)

            # Protein graph data
            batch_protein_graph = Batch.from_data_list(protein_graph_list)
            batch_protein_batch = torch.cat(protein_graph_batch_list, dim=0)

            # LIGAND #
            batch_rdkit_vector = torch.cat(rdkit_vector_list, dim=0)
            batch_roberta_vector = torch.cat(roberta_vector_list, dim=0)

            # Ligand graph data
            batch_ligand_graph = Batch.from_data_list(ligand_graph_list).to(device)
            batch_ligand_batch = torch.cat(ligand_graph_batch_list, dim=0)

            # COMPLEX #
            batch_complex_graph = Batch.from_data_list(complex_graph_list).to(device)
            batch_complex_batch = torch.cat(complex_graph_batch_list, dim=0)

            # GLOBAL
            operator_list = [chr(op[0]) for op in operator_list]

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
                "pdbid": pdbid_list_batch,
            }

            # Initialize array to collect outputs for Monte Carlo samples
            outputs = np.zeros((num_mc_samples, len(batch)))

            for i in range(num_mc_samples):
                # Forward pass with dropout
                output = lightning_model.model(data)
                output_np = output.cpu().numpy().squeeze()  # Shape: (batch_size,)
                outputs[i] = output_np

            # Compute mean and standard deviation for each data point
            mean_preds = np.mean(outputs, axis=0)  # Shape: (batch_size,)
            std_preds = np.std(outputs, axis=0)  # Shape: (batch_size,)

            # Append to predictions and uncertainties lists
            predictions.extend(mean_preds.tolist())
            uncertainties.extend(std_preds.tolist())

            # Collect pdbids
            pdbids.extend(data["pdbid"])

            print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    print("Inference completed.")

    # Create a DataFrame with the results
    results_df = pd.DataFrame(
        {"pdbid": pdbids, "prediction": predictions, "uncertainty": uncertainties}
    )

    return results_df
