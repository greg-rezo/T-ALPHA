import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
from openbabel import pybel
from rdkit import RDLogger
from rdkit import Chem

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
    parser.add_argument("--protein_sequence", default=None)
    parser.add_argument("--smiles", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_ligand_rdkit_mol(sdf_file: Path) -> Chem.Mol:
    suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
    mols = [m for m in suppl if m is not None]
    mol = mols[0]
    if mol is None:
        raise ValueError(f"No valid molecules found in {sdf_file}.")
    return mol


def load_ligand_openbabel_mol(sdf_file: Path) -> pybel.Molecule:
    mol = next(pybel.readfile("sdf", str(sdf_file)))
    if mol is None:
        raise ValueError("Failed to load ligand with OpenBabel.")

    mol.OBMol.AddHydrogens()
    return mol


def main(args) -> np.ndarray:
    RDLogger.DisableLog("rdApp.*")  # Disable RDKit warnings # type: ignore
    warnings.filterwarnings("ignore")

    protein_molecule = next(pybel.readfile("pdb", str(args.protein_file)))
    protein_molecule.OBMol.AddHydrogens()

    # Add hydrogens to the ligand
    ligand_format = args.ligand_file.suffix.lstrip(".")
    assert ligand_format == "sdf", "Currently only support SDF as input files."

    ligand_rdkit_mol = load_ligand_rdkit_mol(args.ligand_file)
    ligand_openbabel_mol = load_ligand_openbabel_mol(args.ligand_file)

    logger.info(f"Loaded protein with {len(protein_molecule.atoms)} atoms.")

    # load model and smiles data files to file cache
    load_t_alpha_files()

    data_list = generate_features(
        protein=protein_molecule,
        openbabel_ligand=ligand_openbabel_mol,
        rdkit_ligand=ligand_rdkit_mol,
        protein_sequence=args.protein_sequence,
        smiles=args.smiles,
    )

    scores = score_data(
        data_list=data_list,
        t_alpha_model_file=DEFAULT_T_ALPHA_MODEL_FILE,
        device=args.device,
    )

    logger.info(f"Scores: {scores}")
    return scores


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s : %(message)s",
    )
    main(args)
