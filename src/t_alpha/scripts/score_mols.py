import argparse
import logging
import warnings
from pathlib import Path

from openbabel import pybel
from rdkit import RDLogger

from t_alpha.data.pipeline import (
    DEFAULT_T_ALPHA_MODEL_FILE,
    generate_features,
    score_data,
    load_t_alpha_files,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_file", type=Path, required=True)
    parser.add_argument("--ligand_file", type=Path, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main(args):
    RDLogger.DisableLog("rdApp.*")  # Disable RDKit warnings # type: ignore
    warnings.filterwarnings("ignore")

    protein_molecule = next(pybel.readfile("pdb", str(args.protein_file)))
    protein_molecule.OBMol.AddHydrogens()

    # Add hydrogens to the ligand
    ligand_format = args.ligand_file.suffix.lstrip(".")
    ligand_mols = []
    for ligand_mol in list(pybel.readfile(ligand_format, str(args.ligand_file))):
        if ligand_mol is None:
            continue
        ligand_mols.append(ligand_mol)

    logger.info(
        f"Loaded protein with {len(protein_molecule.atoms)} atoms and "
        f"{len(ligand_mols)} ligands"
    )

    # load model and smiles data files to cache
    load_t_alpha_files()

    data_list = generate_features(
        protein=protein_molecule,
        ligands=ligand_mols,
    )

    scores = score_data(
        data_list=data_list,
        t_alpha_model_file=DEFAULT_T_ALPHA_MODEL_FILE,
    )

    logger.info(f"Scores: {scores}")


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s : %(message)s",
    )
    main(args)
