import torch


def update_parameter_keys(ckpt_path, output_path):
    """
    Update parameter keys in the checkpoint to handle all mismatched prefixes and names.

    Args:
        ckpt_path (str): Path to the original checkpoint file.
        output_path (str): Path to save the updated checkpoint file.

    Returns:
        None
    """
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Extract the state_dict
    state_dict = checkpoint["state_dict"]

    # Create a new state_dict with renamed keys
    updated_state_dict = {}
    for key, value in state_dict.items():
        # Apply all replacements to handle mismatches
        new_key = key.replace("dmasif_model", "protein_surface_model")
        new_key = new_key.replace("esm_projector", "protein_sequence_projector")
        new_key = new_key.replace(
            "esm_embedding_layer", "protein_sequence_embedding_layer"
        )
        new_key = new_key.replace(
            "rdkit_embedding_layer", "ligand_properties_embedding_layer"
        )
        new_key = new_key.replace("roberta_projector", "ligand_sequence_projector")
        new_key = new_key.replace(
            "roberta_embedding_layer", "ligand_sequence_embedding_layer"
        )
        new_key = new_key.replace(
            "protein_transformer_encoder_layer", "protein_surface_encoder_layer"
        )
        new_key = new_key.replace(
            "protein_transformer_encoder", "protein_surface_encoder"
        )
        new_key = new_key.replace(
            "esm_transformer_decoder_layer", "protein_sequence_decoder_layer"
        )
        new_key = new_key.replace("esm_transformer_decoder", "protein_sequence_decoder")
        new_key = new_key.replace(
            "protein_graph_transformer_decoder_layer", "protein_graph_decoder_layer"
        )
        new_key = new_key.replace(
            "protein_graph_transformer_decoder", "protein_graph_decoder"
        )
        new_key = new_key.replace(
            "ligand_transformer_encoder_layer", "ligand_properties_encoder_layer"
        )
        new_key = new_key.replace(
            "ligand_transformer_encoder", "ligand_properties_encoder"
        )
        new_key = new_key.replace(
            "roberta_transformer_decoder_layer", "ligand_sequence_decoder_layer"
        )
        new_key = new_key.replace(
            "roberta_transformer_decoder", "ligand_sequence_decoder"
        )
        new_key = new_key.replace(
            "ligand_graph_transformer_decoder_layer", "ligand_graph_decoder_layer"
        )
        new_key = new_key.replace(
            "ligand_graph_transformer_decoder", "ligand_graph_decoder"
        )
        new_key = new_key.replace(
            "complex_transformer_encoder_layer", "complex_encoder_layer"
        )
        new_key = new_key.replace("complex_transformer_encoder", "complex_encoder")
        new_key = new_key.replace(
            "protein_transformer_decoder_layer", "protein_decoder_layer"
        )
        new_key = new_key.replace("protein_transformer_decoder", "protein_decoder")
        new_key = new_key.replace(
            "ligand_transformer_decoder_layer", "ligand_decoder_layer"
        )
        new_key = new_key.replace("ligand_transformer_decoder", "ligand_decoder")
        new_key = new_key.replace(
            "complex_graph_transformer_output_embedding_layer",
            "complex_graph_embedding_layer",
        )
        updated_state_dict[new_key] = value

    # Update the checkpoint with the new state_dict
    checkpoint["state_dict"] = updated_state_dict

    # Save the updated checkpoint
    torch.save(checkpoint, output_path)
