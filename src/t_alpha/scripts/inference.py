import argparse

from src.inference.perform_inference import perform_inference
from src.utils.checkpoint_utils import update_parameter_keys


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained model checkpoint."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the checkpoint file (.ckpt).",
    )
    parser.add_argument(
        "--test_set_path",
        type=str,
        required=True,
        help="Path to the test dataset file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for DataLoader. Default: 32.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda' if available.",
    )
    parser.add_argument(
        "--batch_norm", action="store_true", help="Flag to use batch normalization."
    )

    args = parser.parse_args()

    # Update checkpoint keys and overwrite the existing file
    update_parameter_keys(args.ckpt_path, args.ckpt_path)

    # Run inference
    results = perform_inference(
        ckpt_path=args.ckpt_path,
        test_set_path=args.test_set_path,
        batch_size=args.batch_size,
        batch_norm=args.batch_norm,
        device=args.device,
    )

    print("\n--- Inference Complete ---")
    print(f"Checkpoint: {results['checkpoint']}")
    print(f"Epoch: {results['epoch']}")
    print(f"RMSE: {results['RMSE']:.3f}")
    print(f"MAE: {results['MAE']:.3f}")
    print(f"R²: {results['R²']:.3f}")
    print(f"Pearson's r: {results['Pearson r']:.3f}")
    print(f"Spearman's ρ: {results['Spearman ρ']:.3f}")
    print(f"Scatter plot saved to: {results['plot']}")


if __name__ == "__main__":
    main()
