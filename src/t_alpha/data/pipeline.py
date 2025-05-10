"""Data pipeline to generate HDF5 archives."""

import importlib.resources
import logging
import math
import pickle
import functools
import re
import tempfile
from enum import Enum
from pathlib import Path
from typing import ClassVar, Literal, cast, Any
import copy

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from openbabel import openbabel, pybel
from openbabel.pybel import Molecule
from pydantic import BaseModel, ConfigDict
from pykeops.torch import LazyTensor
from pykeops.torch.cluster import grid_cluster
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import (
    AddHs,
    Descriptors,
    Mol,
    RemoveHs,
    SDWriter,
    RemoveStereochemistry,
)
from rdkit.Chem.rdDistGeom import EmbedMolecule, ETKDGv3
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdmolfiles import (
    MolFromSmiles,
    MolToSmiles,
    SDMolSupplier,
)
from rdkit.ML.Descriptors import MoleculeDescriptors  # type: ignore
from sklearn.discriminant_analysis import StandardScaler
from smart_open import open as smart_open
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
import requests

from t_alpha.models.full_model import MetaModel
from t_alpha.training.lightning_module import MetaModelLightning
from t_alpha.utils.checkpoint_utils import update_parameter_keys

SRC_ROOT = Path(__file__).parent.parent
RESOURCES_BASE = "t_alpha.resources"

T_ALPHA_CACHE_DIR = Path.home() / ".cache" / "t_alpha"
DEFAULT_SMILES_TRANSFORMER_MODEL_FILE = (
    T_ALPHA_CACHE_DIR / "SMILES_transformer_params.pt"
)
DEFAULT_T_ALPHA_MODEL_FILE = T_ALPHA_CACHE_DIR / "T-ALPHA_params.ckpt"

logger = logging.getLogger(__name__)


def load_t_alpha_files():
    T_ALPHA_CACHE_DIR.mkdir(exist_ok=True)

    if not DEFAULT_T_ALPHA_MODEL_FILE.exists():
        r = requests.get(
            "https://zenodo.org/records/14514685/files/T-ALPHA_params.ckpt?download=1"
        )
        DEFAULT_T_ALPHA_MODEL_FILE.write_bytes(r.content)

    if not DEFAULT_SMILES_TRANSFORMER_MODEL_FILE.exists():
        r = requests.get(
            "https://zenodo.org/records/14516013/files/Transformer_Encoder_for_SMILES.pt?download=1"
        )
        DEFAULT_SMILES_TRANSFORMER_MODEL_FILE.write_bytes(r.content)


def to_tensor(
    x: np.ndarray,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=dtype).to(device)
    return x.to(device)


class ESMModel(Enum):
    ESM2_T48_15B_UR50D = "esm2_t48_15B_UR50D"
    ESM2_T36_3B_UR50D = "esm2_t36_3B_UR50D"
    ESM2_T36_650M_UR50D = "esm2_t33_650M_UR50D"
    ESM2_T30_150M_UR50D = "esm2_t30_150M_UR50D"


class SMILESTransformerVocab(BaseModel):
    vocab: list[str]
    stoi: dict[str, int]
    block_size: int

    smiles_regex: ClassVar[re.Pattern] = re.compile(
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|<MASK>|<pad>|[CLS]|[EOS])"
    )

    @property
    def itos(self) -> dict[int, str]:
        return {i: ch for i, ch in enumerate(self.vocab)}

    @classmethod
    def from_smiles(
        cls, smiles: list[str], block_size: int = 155
    ) -> "SMILESTransformerVocab":
        special_tokens = {"<MASK>", "<pad>", "[CLS]", "[EOS]"}

        characters = {
            ch for smile in smiles for ch in cls.smiles_regex.findall(smile.strip())
        }
        vocab = sorted(list(special_tokens | characters))
        stoi = {ch: i for i, ch in enumerate(vocab)}

        return cls(
            vocab=vocab,
            stoi=stoi,
            block_size=block_size + 2,
        )


def embed_rdkit_mol(mol: Mol) -> Mol | None:
    """Embed an RDKit molecule."""
    if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
        for i in range(mol.GetNumConformers()):
            mol.RemoveConformer(i)
        mol = RemoveHs(mol)
        mol = AddHs(mol)
        result = EmbedMolecule(mol, ETKDGv3())
        if result == -1:
            return None
        else:
            MMFFOptimizeMolecule(mol)

        return mol


class GraphFeaturizer:
    """
    A class for generating graph-based features for molecular structures,
    including nodes and edges for protein-ligand complexes. The class supports
    various atomic and molecular properties such as van der Waals radii,
    electronegativity, polarizability, and more.

    Attributes:
        surface_features_bool: Indicates whether surface features, including hydrogens, should be computed.
        atom_classes: List of atomic numbers for specific elements (e.g., C, O, N, etc.).
        ATOM_CODES: Mapping of atomic numbers to one-hot encoding indices.
        VDW_RADII: Van der Waals radii for atoms (in pm).
        ELECTRONEGATIVITY: Pauling electronegativity values for atoms.
        POLARIZABILITY: Dipole polarizability values for atoms.
        AMINO_ACID_TYPES: Classification of amino acids by type (e.g., hydrophobic, polar, etc.).
        AMINO_ACID_TYPE_CODES: One-hot encoding for amino acid types.
    """

    # Define the class constructor
    def __init__(self, surface_features_bool: bool = False):
        self.surface_features_bool = surface_features_bool

        # Define the atom classes for C, O, N, S, F, P, Cl, Br, B, I, and other
        self.atom_classes = [
            (6),
            (8),
            (7),
            (16),
            (9),
            (15),
            (17),
            (35),
            (5),
            (53),
            (-1),
        ]

        # add hydrogen if computing surface features
        if surface_features_bool:
            self.atom_classes.insert(0, (1))

        self.ATOM_CODES = {
            atomic_num: idx for idx, atomic_num in enumerate(self.atom_classes)
        }

        # Define the atomic Van der Waals radii (pm) <https://periodic.lanl.gov/89.shtml>
        self.VDW_RADII = {
            5: 192,  # Boron
            6: 170,  # Carbon
            7: 155,  # Nitrogen
            8: 152,  # Oxygen
            9: 135,  # Fluorine
            15: 180,  # Phosphorus
            16: 180,  # Sulfur
            17: 175,  # Chlorine
            35: 183,  # Bromine
            53: 198,  # Iodine
            26: 194,  # Fe (Iron)
            44: 207,  # Ru (Ruthenium)
            34: 190,  # Se (Selenium)
            14: 210,  # Si (Silicon)
            77: 202,  # Ir (Iridium)
            33: 185,  # As (Arsenic)
            27: 192,  # Co (Cobalt)
            23: 179,  # V (Vanadium)
            78: 209,  # Pt (Platinum)
            45: 195,  # Rh (Rhodium)
            4: 153,  # Be (Beryllium)
            76: 216,  # Os (Osmium)
            75: 217,  # Re (Rhenium)
            29: 140,  # Cu (Copper)
            51: 206,  # Sb (Antimony)
            12: 173,  # Mg (Magnesium)
            30: 139,  # Zn (Zinc)
            52: 206,  # Te (Tellurium)
        }

        # Define the electronegativity values as per the Pauling scale
        self.ELECTRONEGATIVITY = {
            5: 2.04,  # Boron
            6: 2.55,  # Carbon
            7: 3.04,  # Nitrogen
            8: 3.44,  # Oxygen
            9: 3.98,  # Fluorine
            15: 2.19,  # Phosphorus
            16: 2.58,  # Sulfur
            17: 3.16,  # Chlorine
            35: 2.96,  # Bromine
            53: 2.66,  # Iodine
            26: 1.83,  # Fe (Iron)
            44: 2.20,  # Ru (Ruthenium)
            34: 2.55,  # Se (Selenium)
            14: 1.90,  # Si (Silicon)
            77: 2.20,  # Ir (Iridium)
            33: 2.18,  # As (Arsenic)
            27: 1.88,  # Co (Cobalt)
            23: 1.63,  # V (Vanadium)
            78: 2.28,  # Pt (Platinum)
            45: 2.28,  # Rh (Rhodium)
            4: 1.57,  # Be (Beryllium)
            76: 2.20,  # Os (Osmium)
            75: 1.90,  # Re (Rhenium)
            29: 1.90,  # Cu (Copper)
            51: 2.05,  # Sb (Antimony)
            12: 1.31,  # Mg (Magnesium)
            30: 1.65,  # Zn (Zinc)
            52: 2.10,  # Te (Tellurium)
        }

        # Define the static dipole polarizability values of neutral elements as defined <https://www.tandfonline.com/doi/full/10.1080/00268976.2018.1535143#d1e161>
        self.POLARIZABILITY = {
            5: 20.50,  # Boron
            6: 11.30,  # Carbon
            7: 7.40,  # Nitrogen
            8: 5.30,  # Oxygen
            9: 3.74,  # Fluorine
            15: 25.10,  # Phosphorus
            16: 19.40,  # Sulfur
            17: 14.60,  # Chlorine
            35: 21.00,  # Bromine
            53: 32.90,  # Iodine
            26: 62.00,  # Fe (Iron)
            44: 72.00,  # Ru (Ruthenium)
            34: 28.90,  # Se (Selenium)
            14: 37.30,  # Si (Silicon)
            77: 54.00,  # Ir (Iridium)
            33: 30.00,  # As (Arsenic)
            27: 55.00,  # Co (Cobalt)
            23: 87.00,  # V (Vanadium)
            78: 48.00,  # Pt (Platinum)
            45: 66.00,  # Rh (Rhodium)
            4: 37.74,  # Be (Beryllium)
            76: 57.00,  # Os (Osmium)
            75: 62.00,  # Re (Rhenium)
            29: 46.50,  # Cu (Copper)
            51: 43.00,  # Sb (Antimony)
            12: 71.20,  # Mg (Magnesium)
            30: 38.67,  # Zn (Zinc)
            52: 38.00,  # Te (Tellurium)
        }

        # Classify each amino acid type as hydrophobic, polar, basic, or acidic
        self.AMINO_ACID_TYPES = {
            "ALA": "hydrophobic",
            "VAL": "hydrophobic",
            "LEU": "hydrophobic",
            "ILE": "hydrophobic",
            "MET": "hydrophobic",
            "PHE": "hydrophobic",
            "TYR": "hydrophobic",
            "TRP": "hydrophobic",
            "SER": "polar",
            "THR": "polar",
            "ASN": "polar",
            "GLN": "polar",
            "LYS": "basic",
            "ARG": "basic",
            "HIS": "basic",
            "ASP": "acidic",
            "GLU": "acidic",
            "CYS": "polar",  # special case
            "GLY": "hydrophobic",  # special case
            "PRO": "hydrophobic",  # special case
            "LIG": "ligand",  # case for ligand atoms
        }

        if surface_features_bool:
            self.VDW_RADII.update({1: 120})  # Hydrogen
            self.ELECTRONEGATIVITY.update({1: 2.20})  # Hydrogen
            self.POLARIZABILITY.update({1: 4.51})  # Hydrogen

        # Define the amino acid type one-hot encodings
        self.AMINO_ACID_TYPE_CODES = {
            "hydrophobic": 0,
            "polar": 1,
            "basic": 2,
            "acidic": 3,
            "ligand": 4,
        }

    # Method to encode the atomic number into a one-hot vector
    def encode_atomic_number(self, atomic_num: int) -> np.ndarray:
        encoding = np.zeros(len(self.atom_classes))
        if atomic_num in self.ATOM_CODES:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        else:
            encoding[self.ATOM_CODES[-1]] = 1.0  # Other class
        return encoding

    # Method to encode amino acid type into a one-hot vector
    def encode_amino_acid_type(
        self, residue_name: str, graph_type: str = "protein"
    ) -> np.ndarray:
        if graph_type == "protein":
            type_str = self.AMINO_ACID_TYPES.get(residue_name, "unknown")
            encoding = np.zeros(len(self.AMINO_ACID_TYPE_CODES) - 1)
            if type_str in self.AMINO_ACID_TYPE_CODES:
                encoding[self.AMINO_ACID_TYPE_CODES[type_str]] = 1.0
            return encoding

        elif graph_type == "complex":
            if residue_name == "LIG":
                type_str = "ligand"
            else:
                type_str = self.AMINO_ACID_TYPES.get(
                    residue_name, "unknown"
                )  # Default to unknown if not found
            encoding = np.zeros(len(self.AMINO_ACID_TYPE_CODES))
            if type_str in self.AMINO_ACID_TYPE_CODES:
                encoding[self.AMINO_ACID_TYPE_CODES[type_str]] = 1.0
            return encoding

        raise ValueError(f"Invalid graph type: {graph_type}")

    # Method to determine if an atom is hydrophobic
    def is_hydrophobic(self, atom: openbabel.OBAtom) -> bool:
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()

        # Check if the atom is neutral
        if formal_charge != 0:
            return False

        # Neutral carbon atoms not bonded to N, O, or F
        if atomic_num == 6:  # Carbon
            bonded_elements = {
                neighbor.GetAtomicNum() for neighbor in openbabel.OBAtomAtomIter(atom)
            }
            if not (
                7 in bonded_elements or 8 in bonded_elements or 9 in bonded_elements
            ):
                return True

        # Neutral sulfur in specific oxidation states (simplified check)
        elif atomic_num == 16:  # Sulfur
            hybridization = atom.GetHyb()
            if hybridization in [
                2,
                3,
            ]:  # Simplified logic for SH or sulfur with sp2/sp3 hybridization
                return True

        # Neutral halogens (Cl, Br, I)
        elif atomic_num in [17, 35, 53]:  # Chlorine, Bromine, Iodine
            return True

        return False

    # Method to calculate the features for a molecule
    def get_node_features(
        self,
        molecule: Molecule,
        source: Literal["ligand", "protein"] = "ligand",
        complex_bool: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        logger.info(f"Generating node features for molecule {source=} {complex_bool=}")

        node_features = []
        coordinates = []

        for i, atom in enumerate(molecule):
            if (
                atom.atomicnum > 1 or self.surface_features_bool
            ):  # Skip hydrogen atoms unless computing surface features
                atomic_number_encoding = self.encode_atomic_number(atom.atomicnum)
                vdw_radius = self.VDW_RADII.get(
                    atom.atomicnum, 183.32
                )  # default to mean value
                formal_charge = atom.OBAtom.GetFormalCharge()
                partial_charge = atom.OBAtom.GetPartialCharge()
                electronegativity = self.ELECTRONEGATIVITY.get(
                    atom.atomicnum, 2.29
                )  # default to mean value
                polarizability = self.POLARIZABILITY.get(
                    atom.atomicnum, 39.12
                )  # default to mean value
                hydrophobic = 1.0 if self.is_hydrophobic(atom.OBAtom) else 0.0
                aromatic = 1.0 if atom.OBAtom.IsAromatic() else 0.0
                acceptor = 1.0 if atom.OBAtom.IsHbondAcceptor() else 0.0
                donor = 1.0 if atom.OBAtom.IsHbondDonor() else 0.0
                ring = 1.0 if atom.OBAtom.IsInRing() else 0.0
                hybridization = atom.OBAtom.GetHyb()
                chirality = 1.0 if atom.OBAtom.IsChiral() else 0.0
                total_degree = len(
                    [neighbor for neighbor in openbabel.OBAtomAtomIter(atom.OBAtom)]
                )
                heavy_degree = sum(
                    1
                    for neighbor in openbabel.OBAtomAtomIter(atom.OBAtom)
                    if neighbor.GetAtomicNum() > 1
                )
                hetero_degree = sum(
                    1
                    for neighbor in openbabel.OBAtomAtomIter(atom.OBAtom)
                    if neighbor.GetAtomicNum() not in [1, 6]
                )
                hydrogen_degree = sum(
                    1
                    for neighbor in openbabel.OBAtomAtomIter(atom.OBAtom)
                    if neighbor.GetAtomicNum() == 1
                )

                atom_coords = np.array(
                    [atom.OBAtom.GetX(), atom.OBAtom.GetY(), atom.OBAtom.GetZ()]
                )
                coordinates.append(atom_coords)

                if complex_bool:
                    source_id = -1 if source == "protein" else 1
                    amino_acid_type_encoding = (
                        self.encode_amino_acid_type(atom.residue.name, "complex")
                        if source == "protein"
                        else self.encode_amino_acid_type("LIG", "complex")
                    )
                    node_features.append(
                        np.concatenate(  # type: ignore
                            (
                                atomic_number_encoding,  # C, O, N, S, F, P, Cl, Br, B, I, other
                                amino_acid_type_encoding,  # hydropobic, polar, basic, acidic, ligand # type: ignore
                                [formal_charge],
                                [hydrophobic],
                                [aromatic],
                                [acceptor],
                                [donor],
                                [ring],
                                [hybridization],
                                [chirality],
                                [total_degree],
                                [heavy_degree],
                                [hetero_degree],
                                [hydrogen_degree],
                                [source_id],
                                [vdw_radius],
                                [partial_charge],
                                [electronegativity],
                                [polarizability],
                            )
                        )
                    )

                else:
                    if source == "protein":
                        amino_acid_type_encoding = self.encode_amino_acid_type(
                            atom.residue.name, "protein"
                        )
                        node_features.append(
                            np.concatenate(  # type: ignore
                                (
                                    atomic_number_encoding,  # C, O, N, S, F, P, Cl, Br, B, I, other
                                    amino_acid_type_encoding,  # hydropobic, polar, basic, acidic # type: ignore
                                    [formal_charge],
                                    [hydrophobic],
                                    [aromatic],
                                    [acceptor],
                                    [donor],
                                    [ring],
                                    [hybridization],
                                    [chirality],
                                    [total_degree],
                                    [heavy_degree],
                                    [hetero_degree],
                                    [hydrogen_degree],
                                    [vdw_radius],
                                    [partial_charge],
                                    [electronegativity],
                                    [polarizability],
                                )
                            )
                        )

                    elif source == "ligand":
                        node_features.append(
                            np.concatenate(
                                (
                                    atomic_number_encoding,  # C, O, N, S, F, P, Cl, Br, B, I, other
                                    [formal_charge],
                                    [hydrophobic],
                                    [aromatic],
                                    [acceptor],
                                    [donor],
                                    [ring],
                                    [hybridization],
                                    [chirality],
                                    [total_degree],
                                    [heavy_degree],
                                    [hetero_degree],
                                    [hydrogen_degree],
                                    [vdw_radius],
                                    [partial_charge],
                                    [electronegativity],
                                    [polarizability],
                                )
                            )
                        )

        # Format and return the features
        node_features = np.array(node_features, dtype=np.float32)
        # print('We converted the node features to a numpy array')
        coordinates = np.array(coordinates, dtype=np.float32)
        # print('We converted the coordinates to a numpy array')
        return node_features, coordinates

    def get_bond_based_edges(self, molecule: Molecule) -> tuple[list, list]:
        logger.info(
            f"Generating bond-based edges for molecule with n_atoms={len(molecule.atoms)}"
        )

        # Remove hydrogens from the molecule
        molecule = safely_remove_hydrogens(molecule)

        edge_idx, edge_attr = [], []
        for bond in openbabel.OBMolBondIter(molecule.OBMol):
            atom1 = (
                bond.GetBeginAtomIdx() - 1
            )  # OpenBabel indices start at 1, so subtract 1 for 0-indexing
            atom2 = bond.GetEndAtomIdx() - 1

            if (
                bond.GetBeginAtom().GetAtomicNum() == 1
                or bond.GetEndAtom().GetAtomicNum() == 1
            ):
                continue  # Skip hydrogen bonds

            bond_order = bond.GetBondOrder()

            atom1_coords = np.array(
                [
                    bond.GetBeginAtom().GetX(),
                    bond.GetBeginAtom().GetY(),
                    bond.GetBeginAtom().GetZ(),
                ]
            )
            atom2_coords = np.array(
                [
                    bond.GetEndAtom().GetX(),
                    bond.GetEndAtom().GetY(),
                    bond.GetEndAtom().GetZ(),
                ]
            )

            distance = np.linalg.norm(atom1_coords - atom2_coords)

            aromatic = 1.0 if bond.IsAromatic() else 0.0
            ring = 1.0 if bond.IsInRing() else 0.0

            atom1_electronegativity = self.ELECTRONEGATIVITY.get(
                bond.GetBeginAtom().GetAtomicNum(), 0
            )
            atom2_electronegativity = self.ELECTRONEGATIVITY.get(
                bond.GetEndAtom().GetAtomicNum(), 0
            )
            electronegativity_difference = np.abs(
                atom1_electronegativity - atom2_electronegativity
            )

            atom1_charge = bond.GetBeginAtom().GetPartialCharge()
            atom2_charge = bond.GetEndAtom().GetPartialCharge()

            electrostatic_energy = (
                atom1_charge * atom2_charge
            ) / distance**2  # TODO: Add this as a feature

            edge_idx.append((atom1, atom2))
            edge_idx.append((atom2, atom1))
            edge_attr.append(
                [
                    bond_order,
                    distance,
                    aromatic,
                    ring,
                    electronegativity_difference,
                    electrostatic_energy,
                ]
            )
            edge_attr.append(
                [
                    bond_order,
                    distance,
                    aromatic,
                    ring,
                    electronegativity_difference,
                    electrostatic_energy,
                ]
            )

        return edge_idx, edge_attr

    def get_distance_based_edges(
        self,
        protein,
        ligand,
        ligand_atom_idx_offset: int,
        distance_threshold: float = 4.5,
    ) -> tuple[
        list, list
    ]:  # 4.5 A corresponds to hydrophobic threshold deined by ProLIF
        # remove hydrogens from the protein and ligand
        protein = safely_remove_hydrogens(protein)
        ligand = safely_remove_hydrogens(ligand)

        # Get the coordinates and electronegativity of the protein atoms
        logger.info("Getting protein coordinates and electronegativity")
        protein_coords, protein_electronegativities, protein_charges = [], [], []
        for i, atom in enumerate(openbabel.OBMolAtomIter(protein.OBMol)):
            assert i == atom.GetIdx() - 1
            if atom.GetAtomicNum() == 1:
                continue  # Skip hydrogen atoms

            protein_coords.append(np.array([atom.GetX(), atom.GetY(), atom.GetZ()]))
            protein_electronegativities.append(
                self.ELECTRONEGATIVITY.get(atom.GetAtomicNum(), 0)
            )
            protein_charges.append(atom.GetPartialCharge())

        # Get the coordinates and electronegativity of the ligand atoms
        logger.info("Getting ligand coordinates and electronegativity")
        ligand_coords, ligand_electronegativities, ligand_charges = [], [], []
        for i, atom in enumerate(openbabel.OBMolAtomIter(ligand.OBMol)):
            assert i == atom.GetIdx() - 1
            if atom.GetAtomicNum() == 1:
                continue  # Skip hydrogen atoms

            ligand_coords.append(np.array([atom.GetX(), atom.GetY(), atom.GetZ()]))
            ligand_electronegativities.append(
                self.ELECTRONEGATIVITY.get(atom.GetAtomicNum(), 0)
            )
            ligand_charges.append(atom.GetPartialCharge())

        protein_coords = np.array(protein_coords)
        ligand_coords = np.array(ligand_coords)

        # Calculate the pairwise distances between protein and ligand atoms
        logger.info("Calculating pairwise distances between protein and ligand atoms")
        distances = np.linalg.norm(
            protein_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :], axis=2
        )
        protein_indices, ligand_indices = np.where(distances <= distance_threshold)

        # Offset ligand atom indices by the number of protein atoms
        logger.info("Generating distance-based edges")
        edge_idx, edge_attr = [], []
        for i, j in zip(protein_indices, ligand_indices):
            assert i < ligand_atom_idx_offset
            ligand_index = j + ligand_atom_idx_offset

            distance = distances[i, j]
            electronegativity_difference = np.abs(
                protein_electronegativities[i] - ligand_electronegativities[j]
            )

            electrostatic_energy = (
                protein_charges[i] * ligand_charges[j]
            ) / distance**2  # TODO: Add this as a feature

            edge_idx.append((i, ligand_index))
            edge_idx.append((ligand_index, i))

            edge_attr.append(
                [0, distance, 0, 0, electronegativity_difference, electrostatic_energy]
            )
            edge_attr.append(
                [0, distance, 0, 0, electronegativity_difference, electrostatic_energy]
            )

        return edge_idx, edge_attr

    def get_protein_ligand_complex_edges(
        self,
        protein: Molecule,
        ligand: Molecule,
        expected_protein_node_count: int | None = None,  # TODO remove
        expected_ligand_node_count: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # preprocess protein and ligand
        protein = protein.clone
        ligand = ligand.clone

        protein = safely_remove_hydrogens(protein)
        ligand = safely_remove_hydrogens(ligand)

        offset = protein.OBMol.NumAtoms()
        logger.debug(f"Found {offset} atoms in protein.")

        # optional validation against node feature counts:
        if expected_protein_node_count:
            assert protein.OBMol.NumAtoms() == expected_protein_node_count

        if expected_ligand_node_count:
            assert ligand.OBMol.NumAtoms() == expected_ligand_node_count

        # Get bond-based edges for protein and ligand
        protein_edges, protein_edge_attrs = self.get_bond_based_edges(protein)
        ligand_edges, ligand_edge_attrs = self.get_bond_based_edges(ligand)

        # Offset ligand atom indices
        ligand_edges = [(i + offset, j + offset) for i, j in ligand_edges]
        assert np.asarray(protein_edges).max() < offset

        # Assign binary interaction labels
        logger.info("Assigning binary interaction labels")
        protein_edge_attrs = [
            (attr + [0]) for attr in protein_edge_attrs
        ]  # 0 for protein-protein
        ligand_edge_attrs = [
            (attr + [0]) for attr in ligand_edge_attrs
        ]  # 0 for ligand-ligand

        # Get distance-based edges between protein and ligand
        protein_ligand_edges, protein_ligand_attrs = self.get_distance_based_edges(
            protein, ligand, ligand_atom_idx_offset=offset
        )
        protein_ligand_attrs = [
            (attr + [1]) for attr in protein_ligand_attrs
        ]  # 1 for protein-ligand

        # Combine all edges
        all_edges = np.array(protein_edges + ligand_edges + protein_ligand_edges)
        all_edge_attrs = np.array(
            protein_edge_attrs + ligand_edge_attrs + protein_ligand_attrs
        )

        return all_edges, all_edge_attrs


class TransformerModel(nn.Module):
    """
    Transformer model for sequence generation.

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Dimension of the token embeddings.
        block_size: Size of the input sequence.
        n_layers: Number of transformer blocks.
        extract_features: Whether to extract features instead of generating output.

    Attributes:
        token_embed: Token embedding layer.
        position_embed: Positional embedding layer.
        dropout: Dropout layer.
        blocks: List of transformer blocks.
        layer_norm: Layer normalization layer.
        output: Output layer.
        extract_features: Whether to extract features instead of generating output.

    Methods:
        forward(idx, admet_props=None, scaffold_idx=None, return_attention_weights=False):
            Forward pass of the model.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        block_size: int = 155,
        n_layers: int = 10,
        extract_features: bool = False,
    ):
        super().__init__()

        self.extract_features = extract_features

        # Define token and position embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Parameter(torch.zeros(1, block_size, embed_dim))

        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList(
            [TransformerBlockModule(embed_dim=embed_dim) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size, bias=True)

    def forward(
        self, idx: torch.Tensor, return_attention_weights: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        # Get batch and time dimensions
        _, T = idx.size()

        # Define token and position embeddings
        token_embeddings = self.token_embed(idx)
        position_embeddings = self.position_embed[:, :T, :]

        # Add token, position, and type embeddings
        x = self.dropout(token_embeddings + position_embeddings)

        # Perform the forward pass through the transformer blocks
        attention_weights_list = []
        for block in self.blocks:
            if return_attention_weights:
                x, attention_weights = block(x, return_attention_weights=True)
                attention_weights_list.append(attention_weights)

            else:
                x, _ = block(x)

        # Apply layer normalization
        x = self.layer_norm(x)

        if self.extract_features:
            cls_embedding = x[:, 0, :]
            return cls_embedding.detach()

        # Generate the output logits
        logits = self.output(x)

        if return_attention_weights:
            return logits, attention_weights_list

        else:
            return logits, None


class TransformerBlockModule(nn.Module):
    """
    Transformer Block class.

    Args:
        embed_dim: The dimensionality of the input embeddings. Default is 256.

    Attributes:
        layer_norm1: Layer normalization module.
        layer_norm2: Layer normalization module.
        attention: Self-attention module.
        mlp: Multi-layer perceptron module.

    Methods:
        forward(x, return_attention_weights=False): Performs forward pass of the transformer block.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()

        # Define layer normalization modules
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        # Define self-attention module
        self.attention = SelfAttentionModule(embed_dim=embed_dim)

        # Define multi-layer perceptron module
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(
        self, x: torch.Tensor, return_attention_weights: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if return_attention_weights:
            y, attention_weights = self.attention(
                self.layer_norm1(x), return_attention_weights=True
            )
            x = x + y
            x = x + self.mlp(self.layer_norm2(x))
            return x, attention_weights

        else:
            y = self.attention(self.layer_norm1(x))
            x = x + y
            x = x + self.mlp(self.layer_norm2(x))
            return x, None


class SelfAttentionModule(nn.Module):
    """
    Self-Attention module that performs attention mechanism on the input sequence.

    Args:
        embed_dim: The dimensionality of the input embeddings.
        n_heads: The number of attention heads.

    Attributes:
        query: Linear layer for computing the query.
        key: Linear layer for computing the key.
        value: Linear layer for computing the value.
        dropout: Dropout layer for regularization.
        projection: Linear layer for projecting the output.
        n_heads: The number of attention heads.

    Methods:
        forward(x, return_attention_weights=False): Performs forward pass of the self-attention module.
    """

    def __init__(self, embed_dim: int = 768, n_heads: int = 12):
        super().__init__()

        # Define linear layers for query, key, and value vectors
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(0.1)
        self.projection = nn.Linear(embed_dim, embed_dim)

        self.n_heads = n_heads

    # Method to perform the forward pass
    def forward(
        self, x: torch.Tensor, return_attention_weights: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Get batch, time, and channel dimensions
        B, T, C = x.size()

        # Compute query, key, and value vectors
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads)
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        scale_factor = 1 / math.sqrt(C // self.n_heads)
        attn_bias = torch.zeros(T, T, dtype=q.dtype, device=q.device)

        # Compute the attention weights
        attn_weight = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.dropout(attn_weight)

        # Multiply the attention weights by the value vectors
        y = torch.matmul(attn_weight, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Map the output to the embedding dimension
        y = self.projection(y)

        y = self.dropout(y)

        if return_attention_weights:
            return y, attn_weight
        else:
            return y


class TransformerFeatureExtractor(torch.nn.Module):
    """
    A class representing a Transformer feature extractor.

    Args:
        model_parameters_file: The path to the model parameters file.
        training_data_file: The path to the training data.
        device: The device to use for computation.

    Attributes:
        device: The device used for computation.
        dataset: The dataset used for training.
        model: The Transformer model used for feature extraction.

    Methods:
        extract_features: Extracts features from input smiles.
    """

    def __init__(
        self,
        model_parameters_file: Path,
        smiles_transformer_vocab: SMILESTransformerVocab,
        device: str,
    ):
        logger.info("Loading transformer feature extractor")
        super().__init__()
        self.device = device

        self.vocab = smiles_transformer_vocab

        # Load the model
        self.model = TransformerModel(
            vocab_size=len(self.vocab.vocab),
            block_size=self.vocab.block_size,
            extract_features=True,
        ).to(self.device)

        # Load the model parameters
        checkpoint = torch.load(model_parameters_file, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        self.model.eval()

    # TODO (philipp): write cache
    def extract_features(self, smiles: str) -> torch.Tensor:
        # Tokenize the SMILES string and pad it to the block size
        smiles = "[CLS]" + smiles.strip() + "[EOS]"
        smiles_tokens = self.vocab.smiles_regex.findall(smiles)
        smiles += "<pad>" * (self.vocab.block_size - len(smiles_tokens))
        token_idx = (
            torch.tensor(
                [self.vocab.stoi[s] for s in self.vocab.smiles_regex.findall(smiles)],
                dtype=torch.long,
            )
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            # Extract the features
            features = self.model(token_idx).squeeze(0)

        return features


class ConnectedGraphFeatureScaler:
    def __init__(self, scaler_file: Path):
        self.node_scaler = _load_scaler(scaler_file, "node_scaler")
        self.edge_scaler = _load_scaler(scaler_file, "edge_scaler")

        self.node_continuous_indices = [
            -4,
            -3,
            -2,
            -1,
        ]  # Last 4 indices for node features
        self.edge_continuous_indices = [
            1,
            4,
            5,
        ]  # Indices 1, 4, and 5 for edge features

    def _standardize_continuous_features(
        self,
        features: np.ndarray,
        continuous_indices: list[int],
        scaler: StandardScaler,
    ) -> np.ndarray:
        # Ensure 'features' is a NumPy array
        features = np.asarray(features)

        # Split continuous and non-continuous features
        continuous_features = features[:, continuous_indices]
        non_continuous_features = np.delete(features, continuous_indices, axis=1)

        # Scale only the continuous features
        standardized_continuous_features = cast(
            np.ndarray, scaler.transform(continuous_features)
        )

        # Concatenate back non-continuous and scaled continuous features
        return np.concatenate(
            [non_continuous_features, standardized_continuous_features],
            axis=1,
        )

    def scale_node_features(self, node_features: np.ndarray) -> np.ndarray:
        return self._standardize_continuous_features(
            node_features, self.node_continuous_indices, self.node_scaler
        )

    def scale_edge_features(self, edge_features: np.ndarray) -> np.ndarray:
        return self._standardize_continuous_features(
            edge_features, self.edge_continuous_indices, self.edge_scaler
        )


class UnconnectedGraphFeatureScaler:
    def __init__(self, scaler_file: Path):
        self.protein_node_scaler = _load_scaler(scaler_file, "protein_node_scaler")

        self.node_continuous_indices = [
            -4,
            -3,
            -2,
            -1,
        ]  # Last 4 indices for node features

    def scale_node_features(self, node_features: np.ndarray) -> np.ndarray:
        continuous_features = node_features[:, self.node_continuous_indices]
        non_continuous_features = np.delete(
            node_features, self.node_continuous_indices, axis=1
        )

        standardized_continuous_features = self.protein_node_scaler.transform(
            continuous_features
        )

        return np.concatenate(
            [non_continuous_features, standardized_continuous_features], axis=1
        )


class LigandFeatures(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    molecule: Molecule
    rdkit_vector: np.ndarray
    transformer_vector: np.ndarray

    coords: np.ndarray
    features: np.ndarray
    edge_ids: np.ndarray
    edge_attrs: np.ndarray

    complex_coords: np.ndarray
    unscaled_complex_features: np.ndarray


class LigandFeatureGenerator:
    ligand_sequence_scaler_name: ClassVar[str] = "ligand_sequence_scaler.pkl"
    ligand_properties_scaler_name: ClassVar[str] = "ligand_properties_scaler.pkl"
    smiles_transformer_training_data_file: ClassVar[Path] = Path(
        "SMILES_transformer_pretraining_data.parquet"
    )
    smiles_transformer_vocab_name: ClassVar[str] = "smiles_transformer_vocab.json.zst"
    connected_graph_scaler_name: ClassVar[str] = "connected_graph_scaler.pkl"

    def __init__(
        self,
        device: str,
        smiles_transformer_model_file: Path = DEFAULT_SMILES_TRANSFORMER_MODEL_FILE,
    ):
        smiles_transformer_vocab = create_or_load_smiles_transformer_vocab(
            self.smiles_transformer_training_data_file,
            self.smiles_transformer_vocab_name,
        )

        self.smiles_transformer_vocab = smiles_transformer_vocab
        self.transformer_feature_extractor = TransformerFeatureExtractor(
            smiles_transformer_vocab=self.smiles_transformer_vocab,
            model_parameters_file=smiles_transformer_model_file,
            device=device,
        )
        self.connected_featurizer = GraphFeaturizer()
        with importlib.resources.path(
            RESOURCES_BASE, self.connected_graph_scaler_name
        ) as connected_graph_scaler_file:
            self.graph_feature_scaler = ConnectedGraphFeatureScaler(
                connected_graph_scaler_file
            )

    @staticmethod
    def _get_rdkit_vector(rdkit_mol: Mol) -> np.ndarray:
        """
        Given an RDKit molecule, compute a 209-dimensional RDKit descriptor vector,
        excluding the descriptor named 'SPS'.

        Steps:
        - Exclude the 'SPS' descriptor from RDKit's default descriptor list.
        - Compute the remaining descriptors using RDKit's MoleculeDescriptorCalculator.
        - Return the resulting vector as a NumPy array.

        Args:
            rdkit_mol: RDKit molecule object.

        Returns:
            A 1D float32 NumPy array of length 209 containing the computed descriptors.
        """
        logger.info("Computing RDKit descriptors")

        # Get all descriptor names, excluding 'SPS'
        all_descriptor_names = [d[0] for d in Descriptors.descList if d[0] != "SPS"]

        # Create a descriptor calculator for the selected descriptors
        RDLogger.DisableLog("rdApp.warning")  # type: ignore
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            all_descriptor_names
        )
        descriptors = calculator.CalcDescriptors(rdkit_mol)
        RDLogger.EnableLog("rdApp.warning")  # type: ignore

        # Convert to NumPy array
        rdkit_vector = np.array(descriptors, dtype=np.float32)

        if len(rdkit_vector) != 209:
            raise ValueError(
                f"Expected a vector of length 209, got {len(rdkit_vector)}."
            )

        if np.isnan(rdkit_vector).any():
            missing_descriptors = [
                desc
                for desc, is_missing in zip(
                    all_descriptor_names, np.isnan(rdkit_vector)
                )
                if is_missing
            ]
            msg = (
                f"Found {len(missing_descriptors)} missing values in RDKit "
                f"feature vector: {missing_descriptors}",
            )
            logger.error(
                msg,
                extra={"missing_descriptors": missing_descriptors},
            )
            raise RuntimeError(msg)

        return rdkit_vector

    @staticmethod
    def _scale_rdkit_vector(rdkit_vector: np.ndarray, scaler_file: Path) -> np.ndarray:
        """
        Given an RDKit descriptor vector and a pre-trained scaler file, this function
        returns the standardized (scaled) version of that vector.

        Args:
            rdkit_vector: The RDKit descriptor vector to be scaled.
                                    Can be 1D or 2D. If 1D, it will be reshaped.
            scaler_file: Path to the pickle file containing the pre-trained scaler
                            dictionary with a key 'rdkit_scaler'.

        Returns:
            The scaled RDKit descriptor vector as a 1D NumPy array.
        """
        logger.info("Scaling RDKit vector")

        # Load the pre-trained scaler
        rdkit_scaler = _load_scaler(scaler_file, "rdkit_scaler")

        # Ensure the vector is 2D for the scaler
        if rdkit_vector.ndim == 1:
            rdkit_vector = rdkit_vector.reshape(1, -1)

        # Scale the vector
        standardized_vector = cast(npt.NDArray, rdkit_scaler.transform(rdkit_vector))

        # Handle NaN values by replacing them with the corresponding mean
        if np.isnan(standardized_vector).any():
            feature_means = cast(
                npt.NDArray, rdkit_scaler.mean_
            )  # Means of the features from the scaler
            standardized_vector = np.where(
                np.isnan(standardized_vector), feature_means, standardized_vector
            )

        # Return as a 1D array if it was originally 1D
        if standardized_vector.shape[0] == 1:
            return standardized_vector.squeeze()
        else:
            return standardized_vector

    @staticmethod
    def _scale_transformer_vector(
        transformer_vector: np.ndarray, scaler_file: Path
    ) -> np.ndarray:
        """
        Given a transformer-based feature embedding and a pre-trained scaler file,
        this function returns the standardized (scaled) version of that embedding.

        Args:
            transformer_vector: The original transformer feature vector to be scaled.
                                            Should be 1D or 2D. If 1D, it will be reshaped.
            scaler_file: Path to the pickle file containing the pre-trained scaler
                            dictionary with a key 'roberta_scaler'.

        Returns:
            The scaled transformer feature embedding as a 1D NumPy array.
        """
        logger.info("Scaling transformer embedding")
        # Load the pre-trained scaler
        roberta_scaler = _load_scaler(scaler_file, "roberta_scaler")

        # Ensure the embedding is 2D for the scaler
        if transformer_vector.ndim == 1:
            transformer_vector = transformer_vector.reshape(1, -1)

        # Scale the vector
        standardized_vector = cast(
            npt.NDArray, roberta_scaler.transform(transformer_vector)
        )

        # Squeeze back to 1D array if applicable
        return standardized_vector.squeeze()

    @staticmethod
    def _get_transformer_vector(
        *,
        transformer_feature_extractor: TransformerFeatureExtractor,
        smiles: str | None = None,
        rdkit_mol: Mol | None = None,
    ) -> np.ndarray:
        """
        Given a ligand RDKit molecule, extract a transformer-based feature vector using a pretrained
        transformer feature extractor.

        Steps:
        - Convert the molecule to a canonical SMILES representation.
        - Pass the canonical SMILES string to the transformer_feature_extractor to obtain the feature vector.
        - Return the feature vector as a NumPy array.

        Args:
            transformer_feature_extractor: An initialized transformer feature
                extractor object with a method
                `extract_features(smiles: str) -> np.ndarray`.
            smiles: The SMILES string. If given, takes precedence over
                extracting the canoncial SMILES from the RDKit mol instance.
            rdkit_mol: RDKit molecule object, used to extract the SMILES string.

        Returns:
            A 1D NumPy array containing the extracted feature vector.
        """
        if smiles is None and rdkit_mol is None:
            raise ValueError("One of smiles and rdkit_mol must be given.")

        logger.info("Extracting transformer features from SMILES")

        if smiles:
            mol = MolFromSmiles(smiles)
            canonical_smiles = MolToSmiles(mol, canonical=True)
        else:
            # Convert to canonical SMILES
            canonical_smiles = MolToSmiles(rdkit_mol, canonical=True)

        # Extract features using the transformer model
        try:
            features = transformer_feature_extractor.extract_features(canonical_smiles)
        except KeyError as e:
            logger.warning(
                f"Failed looking up vocabulary: {e}, trying without stereochemistry."
            )
            # Remove steriochemistry
            # NOTE(Philipp): maybe spend at some point more time here, or re-visit
            # when retraining this part of the model. Some components of SMILES
            # strings did not make it into vocab. I was specifically running into
            # some issues with the `/` and `\` stereochemistry symbols
            rdkit_mol = copy.deepcopy(rdkit_mol)
            RemoveStereochemistry(rdkit_mol)
            canonical_smiles = MolToSmiles(rdkit_mol, canonical=True)
            features = transformer_feature_extractor.extract_features(canonical_smiles)

        # Ensure the result is a NumPy array
        if not isinstance(features, np.ndarray):
            features = np.array(features.cpu(), dtype=np.float32)

        return features

    def from_molecule(
        self,
        molecule: Molecule,
        rdkit_mol: Chem.Mol,
        smiles: str | None = None,
    ) -> "LigandFeatures":
        molecule = safely_remove_hydrogens(molecule)
        rdkit_vector = self._get_rdkit_vector(rdkit_mol)
        transformer_vector = self._get_transformer_vector(
            transformer_feature_extractor=self.transformer_feature_extractor,
            smiles=smiles,
            rdkit_mol=rdkit_mol,
        )

        # Scale the SMILES transformer encoder embedding
        with importlib.resources.path(
            RESOURCES_BASE, self.ligand_sequence_scaler_name
        ) as ligand_sequence_scaler_file:
            scaled_transformer_vector = self._scale_transformer_vector(
                transformer_vector=transformer_vector,
                scaler_file=ligand_sequence_scaler_file,
            )

        # Scale the RDKit 2D descriptor vector
        with importlib.resources.path(
            RESOURCES_BASE, self.ligand_properties_scaler_name
        ) as ligand_properties_scaler_file:
            scaled_rdkit_vector = self._scale_rdkit_vector(
                rdkit_vector=rdkit_vector,
                scaler_file=ligand_properties_scaler_file,
            )

        ligand_node_features, ligand_coords = (
            self.connected_featurizer.get_node_features(
                molecule, source="ligand", complex_bool=False
            )
        )
        ligand_edges, ligand_edge_attrs = map(
            np.array,
            self.connected_featurizer.get_bond_based_edges(molecule),
        )

        complex_features, complex_coords = self.connected_featurizer.get_node_features(
            molecule, source="ligand", complex_bool=True
        )

        scaled_ligand_node_features = self.graph_feature_scaler.scale_node_features(
            ligand_node_features,
        )
        scaled_ligand_edge_attrs = self.graph_feature_scaler.scale_edge_features(
            ligand_edge_attrs,
        )

        return LigandFeatures(
            molecule=molecule,
            rdkit_vector=scaled_rdkit_vector,
            transformer_vector=scaled_transformer_vector,
            features=scaled_ligand_node_features,
            coords=ligand_coords,
            edge_ids=ligand_edges,
            edge_attrs=scaled_ligand_edge_attrs,
            complex_coords=complex_coords,
            unscaled_complex_features=complex_features,
        )


class ProteinFeatures(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    full_molecule: Molecule
    pocket_molecule: Molecule
    sequence: str
    esm2_embedding: np.ndarray

    pocket_coords: np.ndarray
    pocket_features: np.ndarray
    pocket_edge_ids: np.ndarray
    pocket_edge_attrs: np.ndarray

    full_coords: np.ndarray
    unscaled_full_atom_types: np.ndarray
    full_features: np.ndarray

    complex_coords: np.ndarray
    unscaled_complex_features: np.ndarray


class ProteinSequenceScaler:
    def __init__(self, scaler_file: Path):
        self.esm2_scaler = _load_scaler(scaler_file, "esm2_scaler")

    def scale_esm2_embedding(self, esm2_embedding: np.ndarray) -> np.ndarray:
        """
        Given an ESM2 embedding vector and a pre-trained scaler file, this function
        returns the standardized (scaled) version of that embedding.

        Args:
            esm2_embedding: The ESM2 embedding vector to be scaled.
                                        Can be 1D or 2D. If 1D, it will be reshaped.
            scaler_file: Path to the pickle file containing the pre-trained scaler
                            dictionary with a key 'esm2_scaler'.

        Returns:
            The scaled ESM2 embedding as a 1D NumPy array.
        """
        logger.info("Scaling ESM2 embedding")

        # Load the pre-trained scaler

        # Ensure the embedding is 2D for the scaler
        if esm2_embedding.ndim == 1:
            esm2_embedding = esm2_embedding.reshape(1, -1)

        # Scale the embedding
        standardized_embedding = cast(
            npt.NDArray, self.esm2_scaler.transform(esm2_embedding)
        )

        # Return as a 1D array if it was originally 1D
        if standardized_embedding.shape[0] == 1:
            return standardized_embedding.squeeze()
        else:
            return standardized_embedding


class ProteinFeatureGenerator:
    connected_graph_scaler_name: ClassVar[str] = "connected_graph_scaler.pkl"
    unconnected_graph_scaler_name: ClassVar[str] = "unconnected_graph_scaler.pkl"
    protein_sequence_scaler_name: ClassVar[str] = "protein_sequence_scaler.pkl"

    def __init__(self, esm_model_name: ESMModel, device: str, use_cache: bool = True):
        self.esm_model_name = esm_model_name
        self.device = device
        self.connected_featurizer = GraphFeaturizer(surface_features_bool=False)
        self.unconnected_featurizer = GraphFeaturizer(surface_features_bool=True)

        with importlib.resources.path(
            RESOURCES_BASE, self.connected_graph_scaler_name
        ) as connected_graph_scaler_file:
            self.conn_graph_feature_scaler = ConnectedGraphFeatureScaler(
                connected_graph_scaler_file
            )

        with importlib.resources.path(
            RESOURCES_BASE, self.unconnected_graph_scaler_name
        ) as unconnected_graph_scaler_file:
            self.unconn_graph_feature_scaler = UnconnectedGraphFeatureScaler(
                unconnected_graph_scaler_file
            )

        with importlib.resources.path(
            RESOURCES_BASE, self.protein_sequence_scaler_name
        ) as protein_sequence_scaler_file:
            self.protein_sequence_scaler = ProteinSequenceScaler(
                protein_sequence_scaler_file
            )

        logger.info(f"Loading ESM2 model from huggingface: {esm_model_name}")
        # TODO(philipp): pretty sure we are only running ESM2 on CPU currently, enable GPU inference
        if use_cache:
            self.esm_model, self.esm_alphabet, self.esm_batch_converter = (
                load_esm_model_with_cache(self.esm_model_name.value)
            )
        else:
            self.esm_model, self.esm_alphabet, self.esm_batch_converter = (
                load_esm_model(self.esm_model_name.value)
            )

    def __hash__(self) -> int:
        # NOTE: required for making LRU cache working on `_get_esm2_embedding`
        return hash((self.esm_model_name.value, self.device))

    def __eq__(self, other) -> bool:
        # NOTE: required for making LRU cache working on `_get_esm2_embedding`
        return (
            self.esm_model_name == other.esm_model_name and self.device == other.device
        )

    @staticmethod
    def _protein_mol_to_seq(protein_molecule: Molecule) -> str:
        logger.info("Converting protein molecule to sequence")

        with tempfile.TemporaryDirectory() as temp_dir:
            protein_file = Path(temp_dir) / "protein.pdb"
            protein_molecule.write("pdb", str(protein_file))

            # Parse the PDB structure and gather all standard amino acid residues from all chains
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", protein_file)
            assert structure is not None

        # Collect all residues that are standard amino acids from all chains
        # We assume the CSV data was a single continuous sequence, so we replicate that here.
        # Sort chains by chain ID to ensure consistent ordering if needed.
        chains = sorted(structure.get_chains(), key=lambda c: c.id)
        residues = []
        for chain in chains:
            for res in chain.get_residues():
                if res.get_id()[0] == " ":  # standard residue
                    residues.append(res)

        # Convert residues to a single-letter amino acid sequence
        seq = "".join(seq1(res.get_resname()) for res in residues)
        if not seq:
            raise ValueError(
                "No valid amino acid residues found in the provided PDB file."
            )

        return seq

    @functools.lru_cache()
    def _get_esm2_embedding(self, seq: str) -> np.ndarray:
        """Calculate ESM2 embedding vector.

        Given a protein sequence across all protein chains, produce the ESM2
        embedding for that combined sequence.

        This function:
        - Loads the ESM2 model and alphabet as done in the CSV-based code.
        - Converts the sequence into tokens with the model's batch_converter.
        - Runs the ESM2 model to obtain the per-residue representations.
        - Computes the average over residues to obtain a single embedding vector.

        The function has a cache. The cache is shared between different
        instances of ProteinFeatureGenerator as long as the input arguments are
        identical. This is mainly useful in a scenarion when running many
        ligands against a small number of proteins.

        Returns:
        A 1D NumPy array containing the ESM2 embedding vector.

        Requirements:
        - torch
        - biopython
        - numpy
        - The ESM2 model weights will be automatically downloaded by torch.hub if not cached.
        """
        # Convert the single protein sequence into tokens
        logger.info(f"Converting protein sequence of length {len(seq)} into tokens")
        _, _, batch_tokens = self.esm_batch_converter([("1", seq)])
        batch_lens = (batch_tokens != self.esm_alphabet.padding_idx).sum(1)

        # Run the model to get embeddings from layer 33 as done previously
        logger.debug("Running ESM2 model to get embeddings from layer 33")
        self.esm_model.eval()
        with torch.no_grad():
            results = self.esm_model(
                batch_tokens, repr_layers=[33], return_contacts=False
            )
        token_representations = results["representations"][33]

        # Compute the average embedding over all residues
        # Exclude the <cls> and <eos> tokens
        residue_reps = token_representations[0, 1 : batch_lens[0] - 1]
        sequence_representation = residue_reps.mean(0)

        # Return as a NumPy array
        return sequence_representation.cpu().numpy()

    @staticmethod
    def _extract_protein_pocket(
        protein_molecule: Molecule,
        ligand_molecule: Molecule,
        neighbor_radius: float = 8,
    ) -> Molecule:
        """Extract protein pocket from protein-ligand pair.

        The extracted pocket still has hydrogen atoms attached (necessary for
        surface calculations.)
        """
        logger.info("Identifying protein pocket")

        # Load the protein and ligand molecules into biopandas objects
        with tempfile.TemporaryDirectory() as temp_dir:
            protein_file = Path(temp_dir) / "protein.pdb"
            ligand_file = Path(temp_dir) / "ligand.mol2"
            protein_molecule.write("pdb", str(protein_file))
            ligand_molecule.write("mol2", str(ligand_file))

            num_prot_lines = protein_file.read_text().count("\n")
            num_lig_lines = ligand_file.read_text().count("\n")
            logger.info(
                f"Parsing protein with num_lines={num_prot_lines} and"
                f" ligand with num_lines={num_lig_lines}"
            )

            # read in protein pdb file
            protein = PandasPdb().read_pdb(protein_file)

            # read in ligand mol2 file
            ligand = PandasMol2().read_mol2(ligand_file).df
            assert ligand is not None

            logger.info(
                f"Parsed protein with len={len(protein.df)} and"
                f" ligand with len={len(ligand)}"
            )

        # define protein atoms dataframe
        protein_atom = protein.df["ATOM"].reset_index(drop=True)

        # create protein atom dictionary
        protein_atom_dict = protein_atom.to_dict("index")

        # define protein heteroatoms dataframe
        protein_hetatm = protein.df["HETATM"].reset_index(drop=True)

        # create protein heteroatom dictionary
        protein_hetatm_dict = protein_hetatm.to_dict("index")

        # define ligand non-H atoms dataframe
        ligand_nonh = ligand[ligand["atom_type"] != "H"].reset_index(drop=True)

        # create ligand non-H atom dictionary
        ligand_nonh_dict = ligand_nonh.to_dict("index")  # type: ignore

        # initialize lists to save IDs for residues and heteroatoms to keep in pocket file
        pocket_residues = []
        pocket_heteroatoms = []

        # for each protein atom:
        for j in range(len(protein_atom_dict)):
            # if residue number is not already saved in list
            if protein_atom_dict[j]["residue_number"] not in pocket_residues:
                # for each ligand non-H atom:
                for i in range(len(ligand_nonh_dict)):
                    # if Euclidean distance is within 8 Angstroms:
                    if (
                        np.sqrt(
                            (ligand_nonh_dict[i]["x"] - protein_atom_dict[j]["x_coord"])
                            ** 2
                            + (
                                ligand_nonh_dict[i]["y"]
                                - protein_atom_dict[j]["y_coord"]
                            )
                            ** 2
                            + (
                                ligand_nonh_dict[i]["z"]
                                - protein_atom_dict[j]["z_coord"]
                            )
                            ** 2
                        )
                        <= neighbor_radius
                    ):
                        # save chain ID, residue number and insertion to list
                        pocket_residues.append(
                            str(protein_atom_dict[j]["chain_id"])
                            + "_"
                            + str(protein_atom_dict[j]["residue_number"])
                            + "_"
                            + str(protein_atom_dict[j]["insertion"])
                        )

                        break

        # for each protein heteroatom:
        for k in range(len(protein_hetatm_dict)):
            # if residue number is not already saved in list
            if protein_hetatm_dict[k]["residue_number"] not in pocket_heteroatoms:
                # for each ligand non-H atom:
                for i in range(len(ligand_nonh_dict)):
                    # if Euclidean distance is within 8 Angstroms:
                    if (
                        np.sqrt(
                            (
                                ligand_nonh_dict[i]["x"]
                                - protein_hetatm_dict[k]["x_coord"]
                            )
                            ** 2
                            + (
                                ligand_nonh_dict[i]["y"]
                                - protein_hetatm_dict[k]["y_coord"]
                            )
                            ** 2
                            + (
                                ligand_nonh_dict[i]["z"]
                                - protein_hetatm_dict[k]["z_coord"]
                            )
                            ** 2
                        )
                        <= neighbor_radius
                    ):
                        # save heteroatom ID to list
                        pocket_heteroatoms.append(
                            protein_hetatm_dict[k]["residue_number"]
                        )

                        break

        # initialize list to store atoms that are included in saved residues
        atoms_to_keep = []

        # loop through atom dictionary
        for k, v in protein_atom_dict.items():
            # if atom is in saved list
            if (
                str(v["chain_id"])
                + "_"
                + str(v["residue_number"])
                + "_"
                + str(v["insertion"])
                in pocket_residues
            ):
                # append the atom number to new list
                atoms_to_keep.append(v["atom_number"])

        # define the atoms to include in pocket
        residues = protein_atom[(protein_atom["atom_number"].isin(atoms_to_keep))]

        # reset atom number ordering
        residues = residues.reset_index(drop=1)

        # initialize list to store heteroatoms that are included in saved heteroatom IDs
        hetatms_to_keep = []

        # loop through heteroatom dictionary
        for k, v in protein_hetatm_dict.items():
            # if heteroatom is in saved list
            if v["residue_number"] in pocket_heteroatoms and v["residue_name"] == "HOH":
                # append the heteroatom number to new list
                hetatms_to_keep.append(v["atom_number"])

        # define the heteroatoms to include in pocket file
        heteroatoms = protein_hetatm[
            (protein_hetatm["atom_number"].isin(hetatms_to_keep))
        ]

        # reset heteroatom number ordering
        heteroatoms = heteroatoms.reset_index(drop=1)

        # initialize biopandas object to write out pocket pdb file
        pred_pocket = PandasPdb()

        # define the atoms and heteroatoms of the object
        pred_pocket.df["ATOM"], pred_pocket.df["HETATM"] = residues, heteroatoms
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(
                f"Writing pocket pdb file: num_atoms={len(residues)}"
                f" num_hetatoms={len(heteroatoms)}"
            )
            pocket_file = Path(temp_dir) / "protein_pocket.pdb"
            pred_pocket.to_pdb(str(pocket_file))
            pocket_molecule = next(pybel.readfile("pdb", str(pocket_file)))

        return pocket_molecule

    def from_molecule(
        self,
        full_protein_molecule: Molecule,
        ligand_molecule: Molecule,
        sequence: str | None = None,
    ) -> ProteinFeatures:
        if sequence is None:
            sequence = self._protein_mol_to_seq(full_protein_molecule)

        esm2_embedding = self._get_esm2_embedding(sequence)

        pocket_protein_molecule = self._extract_protein_pocket(
            full_protein_molecule.clone, ligand_molecule.clone
        )

        pocket_conn_node_features, pocket_conn_node_coords = (
            self.connected_featurizer.get_node_features(
                pocket_protein_molecule.clone, source="protein", complex_bool=False
            )
        )
        pocket_conn_edge_ids, pocket_conn_edge_attrs = map(
            np.array,
            self.connected_featurizer.get_bond_based_edges(
                pocket_protein_molecule.clone
            ),
        )
        complex_pocket_conn_node_features, complex_pocket_conn_node_coords = (
            self.connected_featurizer.get_node_features(
                pocket_protein_molecule.clone, source="protein", complex_bool=True
            )
        )

        # assert that feature and coord vectors have the expected length
        pocket_protein_molecule_wo_hydrogen = safely_remove_hydrogens(
            pocket_protein_molecule
        )
        expected_pocket_protein_atoms = (
            pocket_protein_molecule_wo_hydrogen.OBMol.NumAtoms()
        )
        expected_pocket_coords = np.asarray(
            [a.coords for a in pocket_protein_molecule_wo_hydrogen]
        )
        assert (
            expected_pocket_protein_atoms
            == pocket_conn_node_features.shape[0]
            == pocket_conn_node_coords.shape[0]
        )
        assert (
            expected_pocket_protein_atoms
            == complex_pocket_conn_node_features.shape[0]
            == complex_pocket_conn_node_coords.shape[0]
        )
        assert np.allclose(
            expected_pocket_coords,
            pocket_conn_node_coords,
        )
        assert np.allclose(
            expected_pocket_coords,
            complex_pocket_conn_node_coords,
        )

        full_unconn_node_features, full_unconn_node_coords = (
            self.unconnected_featurizer.get_node_features(
                full_protein_molecule, source="protein", complex_bool=False
            )
        )
        full_unconn_node_types = full_unconn_node_features[:, :12]  # not scaled
        full_unconn_node_features_only = full_unconn_node_features[
            :, 12:
        ]  # will be scaled

        # Scale features
        scaled_esm2_embedding = self.protein_sequence_scaler.scale_esm2_embedding(
            esm2_embedding
        )

        scaled_pocket_conn_node_features = (
            self.conn_graph_feature_scaler.scale_node_features(
                pocket_conn_node_features,
            )
        )
        scaled_pocket_conn_edge_attrs = (
            self.conn_graph_feature_scaler.scale_edge_features(pocket_conn_edge_attrs)
        )
        scaled_full_unconn_node_features_only = (
            self.unconn_graph_feature_scaler.scale_node_features(
                full_unconn_node_features_only,
            )
        )

        return ProteinFeatures(
            full_molecule=full_protein_molecule,
            pocket_molecule=pocket_protein_molecule,
            sequence=sequence,
            esm2_embedding=scaled_esm2_embedding,
            pocket_coords=pocket_conn_node_coords,
            pocket_features=scaled_pocket_conn_node_features,
            pocket_edge_ids=pocket_conn_edge_ids,
            pocket_edge_attrs=scaled_pocket_conn_edge_attrs,
            full_coords=full_unconn_node_coords,
            unscaled_full_atom_types=full_unconn_node_types,
            full_features=scaled_full_unconn_node_features_only,
            complex_coords=complex_pocket_conn_node_coords,
            unscaled_complex_features=complex_pocket_conn_node_features,
        )


class ComplexFeatures(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    protein_features: ProteinFeatures
    ligand_features: LigandFeatures
    complex_coords: np.ndarray
    complex_features: np.ndarray
    complex_edge_ids: np.ndarray
    complex_edge_attrs: np.ndarray


class ComplexFeatureGenerator:
    connected_graph_scaler_name: ClassVar[str] = "connected_graph_scaler.pkl"

    def __init__(self, device: str):
        self.device = device
        self.connected_featurizer = GraphFeaturizer(surface_features_bool=False)

        with importlib.resources.path(
            RESOURCES_BASE, self.connected_graph_scaler_name
        ) as connected_graph_scaler_file:
            self.conn_graph_feature_scaler = ConnectedGraphFeatureScaler(
                connected_graph_scaler_file
            )

    def from_ligand_protein_features(
        self, ligand_features: LigandFeatures, protein_features: ProteinFeatures
    ) -> ComplexFeatures:
        ligand_node_features, ligand_coords = (
            ligand_features.unscaled_complex_features,
            ligand_features.complex_coords,
        )
        protein_node_features, protein_coords = (
            protein_features.unscaled_complex_features,
            protein_features.complex_coords,
        )
        assert ligand_node_features.shape[0] == ligand_coords.shape[0]
        assert protein_node_features.shape[0] == protein_coords.shape[0]

        protein_molecule = safely_remove_hydrogens(protein_features.pocket_molecule)
        ligand_molecule = safely_remove_hydrogens(ligand_features.molecule)
        assert protein_node_features.shape[0] == protein_molecule.OBMol.NumAtoms()
        assert ligand_node_features.shape[0] == ligand_molecule.OBMol.NumAtoms()

        complex_node_features = np.concatenate(
            (protein_node_features, ligand_node_features), axis=0
        )
        complex_coords = np.concatenate((protein_coords, ligand_coords), axis=0)
        complex_edge_ids, complex_edge_attrs = (
            self.connected_featurizer.get_protein_ligand_complex_edges(
                protein_molecule, ligand_molecule
            )
        )

        scaled_complex_node_features = (
            self.conn_graph_feature_scaler.scale_node_features(
                complex_node_features,
            )
        )
        scaled_complex_edge_attrs = self.conn_graph_feature_scaler.scale_edge_features(
            complex_edge_attrs
        )

        return ComplexFeatures(
            protein_features=protein_features,
            ligand_features=ligand_features,
            complex_coords=complex_coords,
            complex_features=scaled_complex_node_features,
            complex_edge_ids=complex_edge_ids,
            complex_edge_attrs=scaled_complex_edge_attrs,
        )


class ProteinSurfaceFeatures(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    coords: torch.Tensor
    normals: torch.Tensor


class ProteinSurfaceFeatureGenerator:
    def __init__(self, device: str):
        self.device = device

    @staticmethod
    def _atoms_to_points_normals(
        atoms: torch.Tensor,
        batch: torch.Tensor,
        num_atoms: int = 12,
        distance: float = 1.05,
        smoothness: float = 0.5,
        resolution: float = 1.0,
        nits: int = 4,
        atomtypes: torch.Tensor | None = None,
        sup_sampling: int = 20,
        variance: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Turns a collection of atoms into an oriented point cloud.

        Sampling algorithm for protein surfaces, described in Fig. 3 of the paper.

        Args:
            atoms: (N,3) coordinates of the atom centers `a_k`.
            batch: (N,) batch vector, as in PyTorch_geometric.
            distance: value of the level set to sample from
                the smooth distance function. Defaults to 1.05.
            smoothness: radii of the atoms, if atom types are
                not provided. Defaults to 0.5.
            resolution: side length of the cubic cells in
                the final sub-sampling pass. Defaults to 1.0.
            nits: number of iterations . Defaults to 4.
            atomtypes: (N,6) one-hot encoding of the atom
                chemical types. Defaults to None.

        Returns:
            (Tensor): (M,3) coordinates for the surface points `x_i`.
            (Tensor): (M,3) unit normals `n_i`.
            (integer Tensor): (M,) batch vector, as in PyTorch_geometric.
        """
        logger.info("Generating surface point normals")
        # a) Parameters for the soft distance function and its level set:
        T = distance

        N, D = atoms.shape
        B = sup_sampling  # Sup-sampling ratio

        # Batch vectors:
        batch_atoms = batch
        batch_z = batch[:, None].repeat(1, B).view(N * B)

        # b) Draw N*B points at random in the neighborhood of our atoms
        z = atoms[:, None, :] + 10 * T * torch.randn(N, B, D).type_as(atoms)
        z = z.view(-1, D)  # (N*B, D)

        # We don't want to backprop through a full network here!
        atoms = atoms.detach().contiguous()
        z = z.detach().contiguous()

        # N.B.: Test mode disables the autograd engine: we must switch it on explicitely.
        with torch.enable_grad():
            if z.is_leaf:
                z.requires_grad = True

            # c) Iterative loop: gradient descent along the potential
            # ".5 * (dist - T)^2" with respect to the positions z of our samples
            for it in range(nits):
                dists = soft_distances(
                    x=atoms,
                    y=z,
                    batch_x=batch_atoms,
                    batch_y=batch_z,
                    smoothness=smoothness,
                    atomtypes=atomtypes,
                )
                Loss = ((dists - T) ** 2).sum()
                g = torch.autograd.grad(Loss, z)[0]
                z.data -= 0.5 * g

            # d) Only keep the points which are reasonably close to the level set:
            dists = soft_distances(
                atoms,
                z,
                batch_atoms,
                batch_z,
                smoothness=smoothness,
                atomtypes=atomtypes,
            )
            margin = (dists - T).abs()
            mask = margin < variance * T

            # d') And remove the points that are trapped *inside* the protein:
            zz = z.detach()
            zz.requires_grad = True
            for it in range(nits):
                dists = soft_distances(
                    atoms,
                    zz,
                    batch_atoms,
                    batch_z,
                    smoothness=smoothness,
                    atomtypes=atomtypes,
                )
                Loss = (1.0 * dists).sum()
                g = torch.autograd.grad(Loss, zz)[0]
                normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
                zz = zz + 1.0 * T * normals

            dists = soft_distances(
                atoms,
                zz,
                batch_atoms,
                batch_z,
                smoothness=smoothness,
                atomtypes=atomtypes,
            )
            mask = mask & (dists > 1.5 * T)

            z = z[mask].contiguous().detach()
            batch_z = batch_z[mask].contiguous().detach()

            # e) Subsample the point cloud:
            points, batch_points = subsample(z, batch_z, scale=resolution)

            # f) Compute the normals on this smaller point cloud:
            p = points.detach()
            p.requires_grad = True
            dists = soft_distances(
                atoms,
                p,
                batch_atoms,
                batch_points,
                smoothness=smoothness,
                atomtypes=atomtypes,
            )
            Loss = (1.0 * dists).sum()
            g = torch.autograd.grad(Loss, p)[0]
            normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
        points = points - 0.5 * normals
        return points.detach(), normals.detach(), batch_points.detach()

    @staticmethod
    def _select_surface_pocket(
        P_batch: dict[str, torch.Tensor], L_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logger.info("Selecting surface points")
        surface_list = []
        normal_list = []
        batch_list = []

        protein_batch_size = int(P_batch["batch_atoms"][-1].item()) + 1

        for i in range(protein_batch_size):
            P, L = extract_single(P_batch, L_batch, i)
            # Calculate the distance from each protein point to each ligand point
            distances = torch.cdist(
                P["xyz"], L["xyz"], p=2
            )  # Calculate pairwise Euclidean (p=2) distances

            # Find the minimum distance to any ligand point for each protein point
            min_distances, _ = torch.min(distances, dim=1)

            # Create a mask for points within 8 angstroms
            _, indices = torch.sort(min_distances)
            mask = indices[:512]

            # Apply the mask to filter the protein points and their normals
            surface_list.append(P["xyz"][mask])
            normal_list.append(P["normals"][mask])
            batch_list.append(P["batch"][: mask.shape[0]])

        p_xyz = torch.cat(surface_list, dim=0)
        p_normals = torch.cat(normal_list, dim=0)
        p_batch = torch.cat(batch_list, dim=0)

        return p_xyz, p_normals, p_batch

    def from_tensors(
        self,
        atom_coords: torch.Tensor,
        atom_coords_batch: torch.Tensor,
        atom_types: torch.Tensor,
        atom_features: torch.Tensor,
        ligand_coords: torch.Tensor,
        ligand_coords_batch: torch.Tensor,
    ) -> ProteinSurfaceFeatures:
        torch.manual_seed(42)

        P = {}
        L = {}

        # Atom information:
        P["atoms"] = atom_coords.to(self.device)
        P["batch_atoms"] = atom_coords_batch.to(self.device)
        # Chemical features: atom coordinates and types.
        P["atom_xyz"] = atom_coords.to(self.device)
        P["atomtypes"] = atom_types.to(self.device)
        P["atom_features"] = atom_features.to(self.device)

        L["xyz"] = ligand_coords.to(self.device)
        L["batch"] = ligand_coords_batch.to(self.device)

        P["xyz"], P["normals"], P["batch"] = self._atoms_to_points_normals(
            atoms=P["atoms"].to(self.device),
            batch=P["batch_atoms"].to(self.device),
            atomtypes=P["atomtypes"].to(self.device),
        )
        P["xyz"], P["normals"], P["batch"] = self._select_surface_pocket(P, L)

        return ProteinSurfaceFeatures(coords=P["xyz"], normals=P["normals"])


class TAlphaDatasetLoader:
    def __init__(
        self,
        esm_model_name: ESMModel,
        device: str,
        smiles_transformer_model_file: Path = DEFAULT_SMILES_TRANSFORMER_MODEL_FILE,
    ):
        self.device = device
        self.protein_feature_generator = ProteinFeatureGenerator(esm_model_name, device)
        self.ligand_feature_generator = LigandFeatureGenerator(
            device, smiles_transformer_model_file=smiles_transformer_model_file
        )
        self.complex_feature_generator = ComplexFeatureGenerator(device)
        self.protein_surface_feature_generator = ProteinSurfaceFeatureGenerator(device)

    def from_single_protein(
        self,
        protein_molecule: Molecule,
        openbabel_ligand: Molecule,
        rdkit_ligand: Chem.Mol,
        protein_sequence: str | None = None,
        smiles: str | None = None,
    ) -> list[dict]:
        logger.info("Generating protein features...")
        # TODO(philipp): think about whether this could be an issue for
        # evaluating multiple poses/different ligands against a single target.
        # I currently think it is an issue, because we are extracting the pocket
        # based on the molecule and this could be different between different
        # ligands/poses
        protein_features = self.protein_feature_generator.from_molecule(
            protein_molecule, openbabel_ligand, sequence=protein_sequence
        )

        data_list = []
        data = self._collate_data_single_pair(
            protein_features,
            openbabel_ligand,
            rdkit_ligand,
            smiles=smiles,
        )
        data_list.append(data)

        logger.info(f"Generated {len(data_list)} protein-ligand pairs")
        return data_list

    def from_multiple_proteins(
        self,
        protein_molecules: list[Molecule | None],
        openbabel_ligands: list[Molecule | None],
        rdkit_ligands: list[Chem.Mol | None],
        protein_sequences: list[str | None] | None = None,
        smiles_list: list[str | None] | None = None,
    ) -> list[dict | None]:
        """Featurize multiple systems in batch.

        `protein_molecules`, `openbabel_ligands` and `rdkit_ligands` are
        required inputs. The function accepts missing inputs, but will not
        attempt the featurization.

        Args:
          protein_molecules: The OpenBabel protein molecules.
          openbabel_ligands: The OpenBabel ligand molecules.
          rdkit_ligands: The RDKit ligand molecules.
          protein_sequences: The list of protein sequences for the system. While
            this input is optional, it is recommended to give the full protein
            sequence of the system under investigation. For each missing input,
            the sequence will be extracted from the protein molecule, which tends
            to negatively impact performance.
          smiles_list: The list of canonical SMILES strings for the given ligand.
            While this input is optional, it is recommended to pass in the
            expected SMILES string for optimal performance. If not given will be
            extracted from the RDKit molecule.

        Returns:
          List of extracted features. If any of the required inputs are missing
          or if featurization is failing, the respective element in the output
          list is set to None.
        """
        # process inputs
        if protein_sequences is None:
            protein_sequences = [None for _ in range(len(protein_molecules))]
        if smiles_list is None:
            smiles_list = [None for _ in range(len(protein_molecules))]

        # validate matching lengths
        if (
            not len(protein_molecules)
            == len(openbabel_ligands)
            == len(rdkit_ligands)
            == len(protein_sequences)
            == len(smiles_list)
        ):
            raise ValueError("Expected all inputs to be of same length.")

        data_list = []
        for (
            protein_molecule,
            openbabel_ligand,
            rdkit_ligand,
            protein_sequence,
            smiles,
        ) in zip(
            protein_molecules,
            openbabel_ligands,
            rdkit_ligands,
            protein_sequences,
            smiles_list,
        ):
            # robustness to missing inputs
            if (
                protein_molecule is None
                or openbabel_ligand is None
                or rdkit_ligand is None
            ):
                logger.debug("missing inputs, ")
                data_list.append(None)
                continue

            try:
                logger.info(f"Generating protein features for {protein_molecule.title}")
                protein_features = self.protein_feature_generator.from_molecule(
                    protein_molecule, openbabel_ligand, sequence=protein_sequence
                )

                logger.info(f"Generating ligand features for {openbabel_ligand.title}")
                data = self._collate_data_single_pair(
                    protein_features,
                    ligand_molecule=openbabel_ligand,
                    rdkit_ligand_molecule=rdkit_ligand,
                    smiles=smiles,
                )
                data_list.append(data)
            except Exception as e:
                logger.exception(
                    f"Error generating ligand features for protein "
                    f"{protein_molecule.title} and ligand "
                    f"{openbabel_ligand.title}: {e}"
                )
                data_list.append(None)

        logger.info(f"Generated {len(data_list)} protein-ligand pairs")
        return data_list

    def _collate_data_single_pair(
        self,
        protein_features: ProteinFeatures,
        ligand_molecule: Molecule,
        rdkit_ligand_molecule: Chem.Mol,
        smiles: str | None = None,
    ) -> dict:
        atom_coords_batch = torch.zeros(
            protein_features.full_coords.shape[0],  # Changed from size(0)
            dtype=torch.long,
            device=self.device,
        )

        ligand_features = self.ligand_feature_generator.from_molecule(
            ligand_molecule, rdkit_ligand_molecule, smiles=smiles
        )

        complex_features = self.complex_feature_generator.from_ligand_protein_features(
            ligand_features, protein_features
        )

        ligand_graph_data = Data(
            node_feats=to_tensor(ligand_features.features),
            node_coords=to_tensor(ligand_features.coords),
            edge_index=to_tensor(ligand_features.edge_ids, dtype=torch.long)
            .t()
            .contiguous(),
            edge_attr=to_tensor(ligand_features.edge_attrs),
        )
        # TODO: fix batching
        ligand_graph_batch = Batch.from_data_list([ligand_graph_data]).to(  # type: ignore
            self.device
        )
        ligand_graph_batch_id = torch.zeros(
            ligand_graph_data.node_feats.size(0),
            dtype=torch.long,
            device=self.device,
        )

        protein_graph_data = Data(
            node_feats=to_tensor(protein_features.pocket_features),
            node_coords=to_tensor(protein_features.pocket_coords),
            edge_index=to_tensor(protein_features.pocket_edge_ids, dtype=torch.long)
            .t()
            .contiguous(),
            edge_attr=to_tensor(protein_features.pocket_edge_attrs),
        )
        protein_graph_batch = Batch.from_data_list([protein_graph_data]).to(  # type: ignore
            self.device
        )
        protein_graph_batch_id = torch.zeros(
            protein_graph_data.node_feats.size(0),
            dtype=torch.long,
            device=self.device,
        )

        complex_graph_data = Data(
            node_feats=to_tensor(complex_features.complex_features),
            node_coords=to_tensor(complex_features.complex_coords),
            edge_index=to_tensor(complex_features.complex_edge_ids, dtype=torch.long)
            .t()
            .contiguous(),
            edge_attr=to_tensor(complex_features.complex_edge_attrs),
        )
        complex_graph_batch = Batch.from_data_list([complex_graph_data]).to(  # type: ignore
            self.device
        )
        complex_graph_batch_id = torch.zeros(
            complex_graph_data.node_feats.size(0),
            dtype=torch.long,
            device=self.device,
        )

        full_protein_atom_features = to_tensor(
            np.concatenate(
                [
                    protein_features.unscaled_full_atom_types,
                    protein_features.full_features,
                ],
                axis=1,
            )
        )  # shape [N_atoms, D]

        # TODO: move this outside of Dataset
        protein_surface_features = self.protein_surface_feature_generator.from_tensors(
            atom_coords=to_tensor(protein_features.full_coords),
            atom_coords_batch=atom_coords_batch,
            atom_types=to_tensor(protein_features.unscaled_full_atom_types),
            atom_features=full_protein_atom_features,
            ligand_coords=to_tensor(ligand_features.coords),
            ligand_coords_batch=ligand_graph_batch_id,
        )

        surface_batch_id = torch.zeros(
            protein_surface_features.coords.size(0),
            dtype=torch.long,
            device=self.device,
        )

        data = {
            "esm_vector": to_tensor(protein_features.esm2_embedding).unsqueeze(0),
            "rdkit_vector": to_tensor(ligand_features.rdkit_vector).unsqueeze(0),
            "roberta_vector": to_tensor(ligand_features.transformer_vector).unsqueeze(
                0
            ),
            "atom_coords_batch": atom_coords_batch,
            "atom_coords": to_tensor(protein_features.full_coords),
            "atom_features": full_protein_atom_features,
            "surface_coords": protein_surface_features.coords,
            "surface_normals": protein_surface_features.normals,
            "surface_batch_idx": surface_batch_id,
            "protein_graph": protein_graph_batch,
            "protein_graph_batch": protein_graph_batch_id,
            "ligand_graph": ligand_graph_batch,
            "ligand_graph_batch": ligand_graph_batch_id,
            "complex_graph": complex_graph_batch,
            "complex_graph_batch": complex_graph_batch_id,
        }

        return data


def pybel_mol_to_rdkit_mol(mol: Molecule) -> Mol:
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as temp_file:
        # Write the molecule with explicit hydrogens and 3D coordinates
        mol.write("sdf", temp_file.name, overwrite=True)

        # Try to read with RDKit, being more permissive
        suppl = SDMolSupplier(temp_file.name, removeHs=False, sanitize=False)
        mols = [m for m in suppl if m is not None]

        if len(mols) == 0:
            logger.info(f"No valid molecules found in {temp_file.name}, trying SMILES")
            smiles = mol.write("smi").split()[0]  # type: ignore
            rdkit_mol = MolFromSmiles(smiles)
            if rdkit_mol is None:
                raise ValueError(
                    f"No valid molecules found in {temp_file.name} and SMILES conversion failed."
                )
            return rdkit_mol
        elif len(mols) > 1:
            raise ValueError(f"Multiple molecules found in {temp_file.name}.")
        else:
            rdkit_mol = mols[0]

            return rdkit_mol


def pybel_mol_from_rdkit_mol(rdkit_mol: Mol) -> Molecule:
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as temp_file:
        with SDWriter(temp_file.name) as w:
            w.write(rdkit_mol)
        return next(pybel.readfile("sdf", temp_file.name))


def soft_distances(x, y, batch_x, batch_y, smoothness=0.5, atomtypes=None):
    """Computes a soft distance function to the atom centers of a protein.

    Implements Eq. (1) of the paper in a fast and numerically stable way.

    Args:
        x (Tensor): (N,3) atom centers.
        y (Tensor): (M,3) sampling locations.
        batch_x (integer Tensor): (N,) batch vector for x, as in PyTorch_geometric.
        batch_y (integer Tensor): (M,) batch vector for y, as in PyTorch_geometric.
        smoothness (float, optional): atom radii if atom types are not provided. Defaults to .01.
        atomtypes (integer Tensor, optional): (N,6) one-hot encoding of the atom chemical types. Defaults to None.

    Returns:
        Tensor: (M,) values of the soft distance function on the points `y`.
    """
    logger.info("Computing soft distances")
    # Build the (N, M, 1) symbolic matrix of squared distances:
    x_i = LazyTensor(x[:, None, :])  # (N, 1, 3) atoms
    y_j = LazyTensor(y[None, :, :])  # (1, M, 3) sampling points
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M, 1) squared distances

    # Use a block-diagonal sparsity mask to support heterogeneous batch processing:
    D_ij.ranges = diagonal_ranges(batch_x, batch_y)

    if atomtypes is not None:
        atomic_radii = torch.tensor(
            [
                120,  # Hydrogen
                170,  # Carbon
                152,  # Oxygen
                155,  # Nitrogen
                180,  # Sulfur
                135,  # Fluorine
                180,  # Phosphorus
                175,  # Chlorine
                183,  # Bromine
                192,  # Boron
                198,  # Iodine
                190,  # the average vdw radius of the atoms present in the dataset not included in this list
            ],
            dtype=torch.float32,
            device=x.device,
        )

        # normalize radii to min
        atomic_radii = atomic_radii / atomic_radii.min()

        atomtype_radii = atomtypes * atomic_radii[None, :]  # n_atoms, n_atomtypes

        smoothness = torch.sum(
            smoothness * atomtype_radii, dim=1, keepdim=False
        )  # n_atoms, 1
        smoothness_i = LazyTensor(smoothness[:, None, None])

        mean_smoothness = (-D_ij.sqrt()).exp().sum(0)
        mean_smoothness_j = LazyTensor(mean_smoothness[None, :, :])
        mean_smoothness = (
            smoothness_i * (-D_ij.sqrt()).exp() / mean_smoothness_j
        )  # n_atoms, n_points, 1
        mean_smoothness = mean_smoothness.sum(0).view(-1)
        soft_dists = -mean_smoothness * (
            (-D_ij.sqrt() / smoothness_i).logsumexp(dim=0)
        ).view(-1)

    else:
        soft_dists = -smoothness * ((-D_ij.sqrt() / smoothness).logsumexp(dim=0)).view(
            -1
        )

    return soft_dists


def subsample(x, batch=None, scale=1.0):
    """Subsamples the point cloud using a grid (cubic) clustering scheme.

    The function returns one average sample per cell, as described in Fig. 3.e)
    of the paper.

    Args:
        x (Tensor): (N,3) point cloud.
        batch (integer Tensor, optional): (N,) batch vector, as in PyTorch_geometric.
            Defaults to None.
        scale (float, optional): side length of the cubic grid cells. Defaults to 1 (Angstrom).

    Returns:
        (M,3): sub-sampled point cloud, with M <= N.
    """
    logger.info("Subsampling point cloud")
    if batch is None:  # Single protein case:
        labels = grid_cluster(x, scale).long()
        C = labels.max() + 1

        # We append a "1" to the input vectors, in order to
        # compute both the numerator and denominator of the "average"
        #  fraction in one pass through the data.
        x_1 = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
        D = x_1.shape[1]
        points = torch.zeros_like(x_1[:C])
        points.scatter_add_(0, labels[:, None].repeat(1, D), x_1)
        return (points[:, :-1] / points[:, -1:]).contiguous()

    else:
        # comment from dmasif people:

        # We process proteins using a for loop.
        # This is probably sub-optimal, but I don't really know
        # how to do more elegantly (this type of computation is
        # not super well supported by PyTorch).
        batch_size = int(torch.max(batch).item()) + 1  # Typically, =32
        points, batches = [], []
        for b in range(batch_size):
            p = subsample(x[batch == b], scale=scale)
            points.append(p)
            batches.append(b * torch.ones_like(batch[: len(p)]))

    return torch.cat(points, dim=0), torch.cat(batches, dim=0)


def diagonal_ranges(batch_x=None, batch_y=None):
    """
    Encodes the block-diagonal structure associated with batch vectors. This function calculates indices for diagonal blocks (or ranges) and slices for batch vectors.

    Parameters:
        batch_x (torch.Tensor): A tensor representing a batch vector that indicates the batch membership of each item.
        batch_y (torch.Tensor): An optional second tensor representing another batch vector similar to batch_x. If not provided, it is assumed to be the same as batch_x (symmetric case).

    Returns:
        tuple: A tuple containing the diagonal block ranges and slices for both batch vectors, or None if no batches are provided.
    """

    def ranges_slices(batch):
        """
        Helper function to calculate the ranges and slices indices for a given batch vector.

        Parameters:
            batch (torch.Tensor): Batch tensor where each element indicates its batch membership.

        Returns:
            tuple: Tuple containing two elements:
                   - ranges: A tensor of size [num_batches, 2] where each row contains the start and end indices for each batch.
                   - slices: A tensor of indices for each batch, useful for slicing operations.
        """

        # Count the number of elements in each batch
        Ns = batch.bincount()

        # Compute cumulative sum to get the end index for each batch
        indices = Ns.cumsum(0)

        # Append zero at the beginning and combine with indices
        ranges = torch.cat((0 * indices[:1], indices))

        # Create a 2D tensor of [start_idx, end_idx] for each batch
        ranges = (
            torch.stack((ranges[:-1], ranges[1:]))
            .t()
            .int()
            .contiguous()
            .to(batch.device)
        )

        # One-based indices for each batch
        slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

        return ranges, slices

    if batch_x is None and batch_y is None:
        return None  # Exit if no batch data is provided
    elif batch_y is None:
        batch_y = (
            batch_x  # Use batch_x for both if batch_y is not provided (symmetric case)
        )

    ranges_x, slices_x = ranges_slices(
        batch_x
    )  # Calculate ranges and slices for batch_x
    ranges_y, slices_y = ranges_slices(
        batch_y
    )  # Calculate ranges and slices for batch_y

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


def extract_single(P_batch, L_batch, number):
    P = {}
    suface_batch = P_batch["batch"] == number

    P["batch"] = P_batch["batch"][suface_batch]

    # Surface information:
    P["xyz"] = P_batch["xyz"][suface_batch]
    P["normals"] = P_batch["normals"][suface_batch]

    L = {}
    suface_batch = L_batch["batch"] == number

    L["batch"] = L_batch["batch"][suface_batch]

    # Ligand information:
    L["xyz"] = L_batch["xyz"][suface_batch]

    return P, L


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def score_data(
    t_alpha_model_file: Path,
    data_list: list[dict | None],
    batch_size: int = 1,
    device: str | None = None,
) -> np.ndarray:
    assert batch_size == 1, "Batch size must be 1 currently"
    device = device or get_device()

    # Update keys in parameter file
    with tempfile.TemporaryDirectory() as temp_dir:
        # TODO(philipp) write this file to cache to avoid re-running this
        tmp_ckpt_path = Path(temp_dir) / "updated_T-ALPHA_params.ckpt"
        update_parameter_keys(t_alpha_model_file, tmp_ckpt_path)

        logger.info("Loading model from checkpoint")
        model = MetaModel(
            device=device,
            use_protein_graph=True,
            use_protein_surface=True,
            use_protein_sequence=True,
            use_ligand_properties=True,
            use_ligand_graph=True,
            use_ligand_sequence=True,
            use_complex_graph=True,
        )
        lightning_model = MetaModelLightning.load_from_checkpoint(
            train_dataset=None,
            val_dataset=None,
            checkpoint_path=tmp_ckpt_path,
            model=model,
            batch_size=batch_size,
            num_epochs=0,
            batch_norm=True,
        ).to(device)
        lightning_model.eval()

        logger.info("Performing forward pass")
        with torch.no_grad():
            outputs = []
            for data in data_list:
                if data is None:
                    outputs.append(np.nan)
                else:
                    output = lightning_model.model(data)
                    outputs.append(output.cpu().numpy().flatten()[0])

        # output is the predicted binding affinity
        return np.array(outputs)


def embed_pybel_mol(mol: Molecule) -> Molecule | None:
    rdkit_mol = pybel_mol_to_rdkit_mol(mol)
    rdkit_mol = embed_rdkit_mol(rdkit_mol)
    if rdkit_mol is None:
        logger.warning(f"Failed to embed molecule {mol.write('smi')}")
        return None

    return pybel_mol_from_rdkit_mol(rdkit_mol)


def generate_features(
    *,
    protein: Molecule,
    openbabel_ligand: Molecule,
    rdkit_ligand: Chem.Mol,
    protein_sequence: str | None = None,
    smiles: str | None = None,
    esm_model_name: ESMModel = ESMModel.ESM2_T36_3B_UR50D,
    device: str | None = None,
    smiles_transformer_model_file: Path = DEFAULT_SMILES_TRANSFORMER_MODEL_FILE,
) -> list[dict]:
    logger.info("Generating T-ALPHA features...")
    device = device or get_device()

    dataset_loader = TAlphaDatasetLoader(
        esm_model_name=esm_model_name,
        device=device,
        smiles_transformer_model_file=smiles_transformer_model_file,
    )

    return dataset_loader.from_single_protein(
        protein_molecule=protein,
        openbabel_ligand=openbabel_ligand,
        rdkit_ligand=rdkit_ligand,
        protein_sequence=protein_sequence,
        smiles=smiles,
    )


# TODO wrap input into dataclass for simpler logic
def generate_features_in_batch(
    *,
    proteins: list[Molecule | None],
    openbabel_ligands: list[Molecule | None],
    rdkit_ligands: list[Chem.Mol | None],
    protein_sequences: list[str | None] | None = None,
    smiles_list: list[str | None] | None = None,
    esm_model_name: ESMModel = ESMModel.ESM2_T36_3B_UR50D,
    device: str | None = None,
    smiles_transformer_model_file: Path = DEFAULT_SMILES_TRANSFORMER_MODEL_FILE,
) -> list[dict | None]:
    logger.info("Generating T-ALPHA features...")
    device = device or get_device()

    dataset_loader = TAlphaDatasetLoader(
        esm_model_name=esm_model_name,
        device=device,
        smiles_transformer_model_file=smiles_transformer_model_file,
    )

    return dataset_loader.from_multiple_proteins(
        protein_molecules=proteins,
        openbabel_ligands=openbabel_ligands,
        rdkit_ligands=rdkit_ligands,
        protein_sequences=protein_sequences,
        smiles_list=smiles_list,
    )


def create_or_load_smiles_transformer_vocab(
    smiles_transformer_training_data_file: Path,
    smiles_transformer_vocab_name: str,
) -> SMILESTransformerVocab:
    with importlib.resources.path(
        RESOURCES_BASE, smiles_transformer_vocab_name
    ) as smiles_transformer_vocab_file:
        if smiles_transformer_vocab_file.exists():
            logger.info("Loading SMILES vocabulary")
            with smart_open(smiles_transformer_vocab_file, "rt") as f:
                smiles_transformer_vocab = SMILESTransformerVocab.model_validate_json(
                    f.read()
                )
        else:
            logger.info("Generating SMILES vocabulary")
            with smart_open(smiles_transformer_vocab_file, "wt") as f:
                smiles_list = cast(
                    list[str],
                    pd.read_parquet(smiles_transformer_training_data_file)[
                        "SMILES"
                    ].tolist(),
                )

                smiles_transformer_vocab = SMILESTransformerVocab.from_smiles(
                    smiles_list
                )
                f.write(smiles_transformer_vocab.model_dump_json())

    return smiles_transformer_vocab


def safely_remove_hydrogens(mol: Molecule) -> Molecule:
    mol = mol.clone
    n_atoms_before = mol.OBMol.NumAtoms()

    mol.OBMol.DeleteHydrogens()

    h_atoms_remaining = [atom for atom in mol if atom.atomicnum == 1]
    if h_atoms_remaining:
        logger.warning(
            f"Found {len(h_atoms_remaining)} hydrogen atoms after calling "
            f"`DeleteHydrogens`, trying manual deletion..."
        )

        for h_atom in h_atoms_remaining:
            mol.OBMol.DeleteAtom(h_atom.OBAtom)

    if sum(atom.atomicnum == 1 for atom in mol):
        raise ValueError("Manual deletion of hydrogen atoms was unsuccessful.")

    n_atoms_after = mol.OBMol.NumAtoms()

    logger.debug(f"Removed {n_atoms_before - n_atoms_after} hydrogens from molecule.")
    return mol


def load_esm_model(esm_model_name: str) -> tuple[Any, Any, Any]:
    """Load the ESM model and alphabet."""
    esm_model, esm_alphabet = cast(
        tuple[Any, Any], torch.hub.load("facebookresearch/esm:main", esm_model_name)
    )
    esm_batch_converter = esm_alphabet.get_batch_converter()

    return esm_model, esm_alphabet, esm_batch_converter


@functools.cache
def load_esm_model_with_cache(esm_model_name: str) -> tuple[Any, Any, Any]:
    """Load the ESM model and alphabet with in-memory cache."""
    return load_esm_model(esm_model_name)


@functools.cache
def _load_scaler(scaler_file: Path, scaler_name: str) -> StandardScaler:
    # TODO: long term, probably move the scalers to ONNX
    with open(scaler_file, "rb") as f:
        scalers = pickle.load(f)
        return scalers[scaler_name]
