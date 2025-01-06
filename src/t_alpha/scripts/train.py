import argparse

from src.training.train_model import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the MetaModel with configurable options."
    )

    # Required parameters
    parser.add_argument(
        "--train_set",
        type=str,
        required=True,
        help="Path to the training dataset (HDF5).",
    )
    parser.add_argument(
        "--val_set",
        type=str,
        required=True,
        help="Path to the validation dataset (HDF5).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save experiment logs."
    )
    parser.add_argument(
        "--save_name", type=str, required=True, help="Name for the experiment logs."
    )

    # Optional parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training (default: cuda).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=120, help="Total number of training epochs."
    )
    parser.add_argument(
        "--n_epochs", type=int, default=120, help="Number of epochs for training."
    )
    parser.add_argument(
        "--batch_norm",
        type=bool,
        default=True,
        help="Whether to use batch normalization.",
    )
    parser.add_argument(
        "--patience", type=int, default=100, help="Early stopping patience."
    )
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default=None,
        help="Path to a pre-trained model checkpoint.",
    )

    # Feature toggles
    parser.add_argument(
        "--use_protein_graph", action="store_true", help="Use protein graph features."
    )
    parser.add_argument(
        "--use_protein_surface",
        action="store_true",
        help="Use protein surface features.",
    )
    parser.add_argument(
        "--use_protein_sequence",
        action="store_true",
        help="Use protein sequence features.",
    )
    parser.add_argument(
        "--use_ligand_properties", action="store_true", help="Use ligand properties."
    )
    parser.add_argument(
        "--use_ligand_graph", action="store_true", help="Use ligand graph features."
    )
    parser.add_argument(
        "--use_ligand_sequence",
        action="store_true",
        help="Use ligand sequence features.",
    )
    parser.add_argument(
        "--use_complex_graph", action="store_true", help="Use complex graph features."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_model(
        device=args.device,
        train_set=args.train_set,
        val_set=args.val_set,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        batch_norm=args.batch_norm,
        patience=args.patience,
        n_epochs=args.n_epochs,
        checkpoint_dir=args.checkpoint_dir,
        save_dir=args.save_dir,
        save_name=args.save_name,
        model_checkpoint_path=args.model_checkpoint_path,
        use_protein_graph=args.use_protein_graph,
        use_protein_surface=args.use_protein_surface,
        use_protein_sequence=args.use_protein_sequence,
        use_ligand_properties=args.use_ligand_properties,
        use_ligand_graph=args.use_ligand_graph,
        use_ligand_sequence=args.use_ligand_sequence,
        use_complex_graph=args.use_complex_graph,
    )
