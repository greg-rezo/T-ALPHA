import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataListLoader

from t_alpha.data.full_model_dataset import MetaModelDataset
from t_alpha.models.full_model import MetaModel
from t_alpha.training.lightning_module import MetaModelLightning


def perform_inference(
    ckpt_path: str,
    test_set_path: str,
    batch_size: int = 32,
    batch_norm: bool = True,
    device: str = "cpu",
):
    """
    Perform inference.

    Args:
        ckpt_path (str): Path to the checkpoint file (.ckpt).
        test_set_path (str): Path to the test dataset.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to True.
        device (str, optional): Device to run the inference on. Defaults to 'cuda' if available.

    Returns:
        dict: A dictionary containing the inference metrics and the path to the saved plot.
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
        num_epochs=0,
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

    predictions = []
    targets = []
    pdbids = []

    with torch.no_grad():
        for batch in test_loader:
            # Copy the data processing logic from your process_batch method
            # Ensure the batch is processed exactly as during training

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
                label_list.append(data_item.label)
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
                "pdbid": pdbid_list_batch,
            }

            # Forward pass
            output = lightning_model.model(data)

            # Collect predictions and targets
            predictions.extend(output.cpu().numpy())
            targets.extend(data["label"].cpu().numpy())
            pdbids.extend(data["pdbid"])  # Collect IDs if needed

    # Create a DataFrame with predictions and targets
    df = pd.DataFrame(
        {
            "pdbid": pdbids,
            "prediction": [pred[0] for pred in predictions],
            "target": [tgt[0] for tgt in targets],
        }
    )

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(df["target"], df["prediction"]))
    mae = mean_absolute_error(df["target"], df["prediction"])
    r2 = r2_score(df["target"], df["prediction"])
    pearson_corr, _ = pearsonr(df["prediction"], df["target"])
    spearman_corr, _ = spearmanr(df["prediction"], df["target"])

    # Print the results for the checkpoint
    print(f"Checkpoint: {os.path.basename(checkpoint_file)}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"Pearson's r: {pearson_corr:.3f}")
    print(f"Spearman's ρ: {spearman_corr:.3f}")

    # Plot scatter plot for predictions vs. targets
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["target"],
        df["prediction"],
        alpha=0.6,
        color="steelblue",
        edgecolors="none",
        s=20,
    )
    plt.plot(
        [df["target"].min(), df["target"].max()],
        [df["target"].min(), df["target"].max()],
        lw=1.5,
        color="black",
        ls="--",
    )
    plt.xlabel("True Values", fontsize=18)
    plt.ylabel("Predicted Values", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(False)

    # Add metrics to the plot
    metrics_text = (
        f"RMSE: {rmse:.3f}\n"
        f"MAE: {mae:.3f}\n"
        f"$r^2$: {r2:.3f}\n"
        f"Pearson $r$: {pearson_corr:.3f}\n"
        f"Spearman $\\rho$: {spearman_corr:.3f}"
    )
    plt.text(
        0.35,
        0.95,
        metrics_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=14,
        bbox=dict(facecolor="white", alpha=0.5),
        horizontalalignment="right",
    )

    plt.savefig("scatter_plot.png")  # Save the plot
    plt.show()

    # Return the metrics and plot filename
    return {
        "checkpoint": os.path.basename(checkpoint_file),
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "Pearson r": pearson_corr,
        "Spearman ρ": spearman_corr,
        "plot": "scatter_plot.png",
    }
