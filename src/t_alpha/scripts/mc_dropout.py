import argparse

import torch

from t_alpha.inference.mc_dropout import perform_mc_dropout
from t_alpha.utils.uncertainty_weighting import apply_uncertainty_weighting


def main(args):
    # Perform MC Dropout inference
    print("Performing MC Dropout...")
    results_df = perform_mc_dropout(
        ckpt_path=args.checkpoint,
        test_set_path=args.test_set,
        batch_size=args.batch_size,
        batch_norm=args.batch_norm,
        device=args.device,
        num_mc_samples=args.num_mc_samples,
    )

    # Save MC Dropout results
    if args.output_results:
        results_df.to_csv(args.output_results, index=False)
        print(f"MC Dropout results saved to {args.output_results}")

    # Apply uncertainty weighting
    print("Applying uncertainty weighting...")
    weighted_df = apply_uncertainty_weighting(
        df=results_df, scale=args.scale, output_csv=args.output_weighted
    )

    if args.output_weighted:
        print(f"Weighted results saved to {args.output_weighted}")

    print("MC Dropout with uncertainty weighting completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MC Dropout inference and apply uncertainty weighting."
    )

    # Arguments for MC Dropout
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file (.ckpt).",
    )
    parser.add_argument(
        "--test_set", type=str, required=True, help="Path to the test dataset."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference."
    )
    parser.add_argument(
        "--batch_norm", type=bool, default=True, help="Use batch normalization."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--num_mc_samples", type=int, default=100, help="Number of Monte Carlo samples."
    )

    # Arguments for uncertainty weighting
    parser.add_argument(
        "--scale",
        type=float,
        default=10,
        help="Scale parameter for uncertainty weighting.",
    )
    parser.add_argument(
        "--output_results", type=str, help="Path to save MC Dropout results as CSV."
    )
    parser.add_argument(
        "--output_weighted", type=str, help="Path to save weighted results as CSV."
    )

    args = parser.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
