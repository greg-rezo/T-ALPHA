import argparse
import logging
import warnings
from pathlib import Path

from rdkit import RDLogger

from t_alpha.data.pipeline import ESMModel, run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_file", type=Path, required=True)
    parser.add_argument("--ligand_file", type=Path, required=True)
    parser.add_argument(
        "--esm_model", type=ESMModel, default=ESMModel.ESM2_T36_650M_UR50D
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main(args):
    RDLogger.DisableLog("rdApp.*")  # Disable RDKit warnings # type: ignore
    warnings.filterwarnings("ignore")
    run(
        protein_file=args.protein_file,
        ligand_file=args.ligand_file,
        esm_model_name=args.esm_model,
        calc_surface_features=True,
    )


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s : %(message)s",
    )
    main(args)
