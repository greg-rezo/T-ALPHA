"""Data pipeline to generate HDF5 archives."""

import importlib.resources
import logging
import math
import pickle
import re
import tempfile
from enum import Enum
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from openbabel import openbabel, pybel
from openbabel.pybel import Molecule
from pydantic import BaseModel
from pykeops.torch import LazyTensor
from pykeops.torch.cluster import grid_cluster
from rdkit.Chem import Descriptors, Mol
from rdkit.Chem.rdmolfiles import (
    MolToSmiles,
    SDMolSupplier,
)
from rdkit.ML.Descriptors import MoleculeDescriptors  # type: ignore
from sklearn.discriminant_analysis import StandardScaler
from smart_open import open as smart_open
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from t_alpha.models.full_model import MetaModel
from t_alpha.training.lightning_module import MetaModelLightning
from t_alpha.utils.checkpoint_utils import update_parameter_keys

SRC_ROOT = Path(__file__).parent.parent
RESOURCES_BASE = "t_alpha.resources"

logger = logging.getLogger(__name__)


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


# class TransformerDataset(Dataset):
#     """
#     Dataset class for the Transformer model.

#     Args:
#         data_file: The path to the data file.
#         block_size: The block size for padding the sequences.

#     Attributes:
#         smiles: The SMILES data from the dataset.
#         smiles_regex: The regular expression pattern for tokenizing SMILES strings.
#         vocab: The vocabulary of special tokens and characters.
#         stoi: A mapping of characters to their corresponding indices in the vocabulary.
#         itos: A mapping of indices to their corresponding characters in the vocabulary.
#         block_size: The block size for padding the sequences.

#     Methods:
#         __len__(): Returns the length of the dataset.
#         __getitem__(idx): Returns the item at the given index.

#     """

#     def __init__(self, data_file: Path, block_size: int = 155):
#         # Retrieve the SMILES strings from the data file
#         data = pd.read_parquet(data_file)
#         self.smiles = data["SMILES"]

#         # Tokenize the SMILES strings
#         self.smiles_regex = re.compile(
#             r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|<MASK>|<pad>|[CLS]|[EOS])"
#         )
#         special_tokens = {"<MASK>", "<pad>", "[CLS]", "[EOS]"}

#         # # Build the vocabulary
#         characters = {
#             ch
#             for smile in self.smiles
#             for ch in self.smiles_regex.findall(smile.strip())
#         }
#         self.vocab = sorted(list(special_tokens | characters))

#         # Create mappings for the vocabulary
#         self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
#         self.itos = {i: ch for i, ch in enumerate(self.vocab)}

#         self.block_size = 2 + block_size

#     # Method to return the length of the dataset
#     def __len__(self) -> int:
#         return len(self.smiles)

#     # Method to return the item at the given index
#     def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         # Tokenize the SMILES string and pad it to the block size
#         smiles = "[CLS]" + self.smiles[idx].strip() + "[EOS]"
#         smiles_tokens = self.smiles_regex.findall(smiles)
#         smiles += "<pad>" * (self.block_size - len(smiles_tokens))

#         # Retrieve the token indices
#         true_token_idx = [self.stoi[s] for s in self.smiles_regex.findall(smiles)]

#         # Mask the tokens
#         mask_idx = []
#         for s in range(len(smiles_tokens)):
#             if random.random() < 0.15:
#                 mask_idx.append(False)
#                 num = random.random()
#                 if num >= 0.2:
#                     smiles_tokens[s] = "<MASK>"
#                 elif num >= 0.1:
#                     smiles_tokens[s] = self.vocab[
#                         int(random.random() * len(self.vocab))
#                     ]
#             else:
#                 mask_idx.append(True)

#         # Identify the masked tokens
#         mask_idx += [True] * (self.block_size - len(mask_idx))
#         masked_smiles = "".join(smiles_tokens)

#         # Pad to the block size
#         masked_smiles += "<pad>" * (
#             self.block_size - len(self.smiles_regex.findall(masked_smiles))
#         )
#         masked_token_idx = [
#             self.stoi[s] for s in self.smiles_regex.findall(masked_smiles)
#         ]

#         return (
#             torch.tensor(masked_token_idx, dtype=torch.long),
#             torch.tensor(true_token_idx, dtype=torch.long),
#             torch.tensor(mask_idx),
#         )


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


# Define a class for the transformer feature extractor
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


def _add_hydrogens_and_save(molecule: Molecule, output_file: Path, output_format: str):
    """Add hydrogens to the given molecule and save it in the specified format, overwriting if needed."""
    molecule.OBMol.AddHydrogens()
    output_file_obj = pybel.Outputfile(
        output_format, str(output_file), overwrite=True
    )  # Overwrite existing files
    output_file_obj.write(molecule)
    output_file_obj.close()


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
        raise ValueError("No valid amino acid residues found in the provided PDB file.")

    return seq


def _get_esm2_embedding(seq: str, esm_model_name: ESMModel) -> np.ndarray:
    """
    Given a protein sequence across all protein chains, produce the ESM2 embedding for
    that combined sequence.

    This function:
      - Loads the ESM2 model and alphabet as done in the CSV-based code.
      - Converts the sequence into tokens with the model's batch_converter.
      - Runs the ESM2 model to obtain the per-residue representations.
      - Computes the average over residues to obtain a single embedding vector.

    Returns:
      A 1D NumPy array containing the ESM2 embedding vector.

    Requirements:
      - torch
      - biopython
      - numpy
      - The ESM2 model weights will be automatically downloaded by torch.hub if not cached.
    """

    logger.info(f"Loading ESM2 model from huggingface: {esm_model_name}")
    esm_model, esm_alphabet = torch.hub.load(
        "facebookresearch/esm:main", esm_model_name.value
    )

    # Convert the single protein sequence into tokens
    logger.info("Converting protein sequence into tokens")
    esm_batch_converter = esm_alphabet.get_batch_converter()
    _, _, batch_tokens = esm_batch_converter([("1", seq)])
    batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)

    # Run the model to get embeddings from layer 33 as done previously
    logger.info("Running ESM2 model to get embeddings from layer 33")
    esm_model.eval()
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    # Compute the average embedding over all residues
    # Exclude the <cls> and <eos> tokens
    residue_reps = token_representations[0, 1 : batch_lens[0] - 1]
    sequence_representation = residue_reps.mean(0)

    # Return as a NumPy array
    return sequence_representation.cpu().numpy()


def _pybel_mol_to_rdkit_mol(mol: Molecule) -> Mol:
    with tempfile.TemporaryDirectory() as temp_dir:
        sdf_file = Path(temp_dir) / "mol.sdf"
        mol.write("sdf", str(sdf_file))

        # Load the molecule into RDKit
        suppl = SDMolSupplier(str(sdf_file), removeHs=False)
        mols = [m for m in suppl if m is not None]
        if len(mols) == 0:
            raise ValueError(f"No valid molecules found in {sdf_file}.")
        elif len(mols) > 1:
            raise ValueError(f"Multiple molecules found in {sdf_file}.")
        else:
            mol = mols[0]

    return mol


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
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(all_descriptor_names)

    # Compute the descriptors
    descriptors = calculator.CalcDescriptors(rdkit_mol)

    # Convert to NumPy array
    rdkit_vector = np.array(descriptors, dtype=np.float32)

    if len(rdkit_vector) != 209:
        raise ValueError(f"Expected a vector of length 209, got {len(rdkit_vector)}.")

    return rdkit_vector


def _get_transformer_vector(
    rdkit_mol: Mol, transformer_feature_extractor: TransformerFeatureExtractor
) -> np.ndarray:
    """
    Given a ligand RDKit molecule, extract a transformer-based feature vector using a pretrained
    transformer feature extractor.

    Steps:
      - Convert the molecule to a canonical SMILES representation.
      - Pass the canonical SMILES string to the transformer_feature_extractor to obtain the feature vector.
      - Return the feature vector as a NumPy array.

    Args:
        rdkit_mol: RDKit molecule object.
        transformer_feature_extractor: An initialized transformer feature extractor object
                                       with a method `extract_features(smiles: str) -> np.ndarray`.

    Returns:
        A 1D NumPy array containing the extracted feature vector.
    """
    logger.info("Extracting transformer features from SMILES")

    # Convert to canonical SMILES
    canonical_smiles = MolToSmiles(rdkit_mol, canonical=True)

    # Extract features using the transformer model
    features = transformer_feature_extractor.extract_features(canonical_smiles)

    # Ensure the result is a NumPy array
    if not isinstance(features, np.ndarray):
        features = np.array(features.cpu(), dtype=np.float32)

    return features


def _extract_protein_pocket(
    protein_molecule: Molecule, ligand_molecule: Molecule, neighbor_radius: float = 8
) -> Molecule:
    logger.info("Identifying protein pocket")

    # Load the protein and ligand molecules into biopandas objects
    with tempfile.TemporaryDirectory() as temp_dir:
        protein_file = Path(temp_dir) / "protein.pdb"
        ligand_file = Path(temp_dir) / "ligand.mol2"
        protein_molecule.write("pdb", str(protein_file))
        ligand_molecule.write("mol2", str(ligand_file))

        # read in protein pdb file
        protein = PandasPdb().read_pdb(protein_file)

        # read in ligand mol2 file
        ligand = PandasMol2().read_mol2(ligand_file).df
        assert ligand is not None

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
    ligand_nonh_dict = ligand_nonh.to_dict("index")

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
                        + (ligand_nonh_dict[i]["y"] - protein_atom_dict[j]["y_coord"])
                        ** 2
                        + (ligand_nonh_dict[i]["z"] - protein_atom_dict[j]["z_coord"])
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
                        (ligand_nonh_dict[i]["x"] - protein_hetatm_dict[k]["x_coord"])
                        ** 2
                        + (ligand_nonh_dict[i]["y"] - protein_hetatm_dict[k]["y_coord"])
                        ** 2
                        + (ligand_nonh_dict[i]["z"] - protein_hetatm_dict[k]["z_coord"])
                        ** 2
                    )
                    <= neighbor_radius
                ):
                    # save heteroatom ID to list
                    pocket_heteroatoms.append(protein_hetatm_dict[k]["residue_number"])

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
    heteroatoms = protein_hetatm[(protein_hetatm["atom_number"].isin(hetatms_to_keep))]

    # reset heteroatom number ordering
    heteroatoms = heteroatoms.reset_index(drop=1)

    # initialize biopandas object to write out pocket pdb file
    pred_pocket = PandasPdb()

    # define the atoms and heteroatoms of the object
    pred_pocket.df["ATOM"], pred_pocket.df["HETATM"] = residues, heteroatoms
    with tempfile.TemporaryDirectory() as temp_dir:
        pocket_file = Path(temp_dir) / "protein_pocket.pdb"
        pred_pocket.to_pdb(str(pocket_file))
        pocket_molecule = next(pybel.readfile("pdb", str(pocket_file)))

    return pocket_molecule


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
        source: str = "ligand",
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
        molecule.OBMol.DeleteHydrogens()

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
        self, protein, ligand, distance_threshold=4.5
    ) -> tuple[
        list, list
    ]:  # 4.5 A corresponds to hydrophobic threshold deined by ProLIF
        # remove hydrogens from the protein and ligand
        protein.OBMol.DeleteHydrogens()
        ligand.OBMol.DeleteHydrogens()

        # Get the coordinates and electronegativity of the protein atoms
        logger.info("Getting protein coordinates and electronegativity")
        protein_coords, protein_electronegativities, protein_charges = [], [], []
        for atom in openbabel.OBMolAtomIter(protein.OBMol):
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
        for atom in openbabel.OBMolAtomIter(ligand.OBMol):
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
            ligand_index = j + len(protein_coords)

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
        self, protein, ligand
    ) -> tuple[np.ndarray, np.ndarray]:
        # Get bond-based edges for protein and ligand
        protein_edges, protein_edge_attrs = self.get_bond_based_edges(protein)
        ligand_edges, ligand_edge_attrs = self.get_bond_based_edges(ligand)

        # Offset ligand atom indices
        ligand_edges_offset = [
            (i + protein.OBMol.NumAtoms(), j + protein.OBMol.NumAtoms())
            for i, j in ligand_edges
        ]

        # Assign binary interaction labels
        logger.info("Assigning binary interaction labels")
        protein_edge_attrs = [
            attr + [0] for attr in protein_edge_attrs
        ]  # 0 for protein-protein
        ligand_edge_attrs = [
            attr + [0] for attr in ligand_edge_attrs
        ]  # 0 for ligand-ligand

        # Get distance-based edges between protein and ligand
        protein_ligand_edges, protein_ligand_attrs = self.get_distance_based_edges(
            protein, ligand
        )
        protein_ligand_attrs = [
            attr + [1] for attr in protein_ligand_attrs
        ]  # 1 for protein-ligand

        # Combine all edges
        all_edges = np.array(protein_edges + ligand_edges_offset + protein_ligand_edges)
        all_edge_attrs = np.array(
            protein_edge_attrs + ligand_edge_attrs + protein_ligand_attrs
        )

        return all_edges, all_edge_attrs


def _get_graphs(
    connected_featurizer: GraphFeaturizer,
    unconnected_featurizer: GraphFeaturizer,
    prot: Molecule,
    lig: Molecule,
    full_prot_withH: Molecule,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Generates graph-based features and attributes for a protein pocket, ligand,
    protein-ligand complex, and unconnected protein.

    Args:
        connected_featurizer: An instance of the `GraphFeaturizer`
            class for generating connected graph features (e.g., bonds and interactions).
        unconnected_featurizer: An instance of the `GraphFeaturizer`
            class for generating unconnected features (e.g., atomic-level features only).
        prot: The protein pocket structure.
        lig: The ligand structure.
        full_prot_withH: The full protein structure with hydrogens.

    Returns:
        - protein_coords Coordinates of protein pocket atoms.
        - protein_features Features of protein pocket atoms.
        - prot_edges Bond-based edges for the protein pocket.
        - prot_attrs Attributes of bond-based edges for the protein pocket.
        - ligand_coords Coordinates of ligand atoms.
        - ligand_features Features of ligand atoms.
        - lig_edges Bond-based edges for the ligand.
        - lig_attrs Attributes of bond-based edges for the ligand.
        - complex_coords Coordinates of protein-ligand complex atoms.
        - complex_features Features of protein-ligand complex atoms.
        - complex_edges Edges (both bond-based and distance-based)
            for the protein-ligand complex.
        - complex_attrs Attributes of edges for the protein-ligand complex.
        - prot_withH_coords Coordinates of protein atoms (with hydrogens).
        - prot_withH_atom_types One-hot encoded atom types for the protein
            (with hydrogens).
        - prot_withH_features Features of protein atoms (excluding atom types)
            for the protein (with hydrogens).
    """
    logger.info("Getting graph features")

    # Process protein pocket
    protein_features, protein_coords = connected_featurizer.get_node_features(
        prot, source="protein", complex_bool=False
    )
    prot_edges, prot_attrs = connected_featurizer.get_bond_based_edges(prot)

    # Process ligand
    ligand_features, ligand_coords = connected_featurizer.get_node_features(
        lig, source="ligand", complex_bool=False
    )
    lig_edges, lig_attrs = connected_featurizer.get_bond_based_edges(lig)

    # Process protein-ligand complex
    protein_features_complex, protein_coords_complex = (
        connected_featurizer.get_node_features(
            prot, source="protein", complex_bool=True
        )
    )
    ligand_features_complex, ligand_coords_complex = (
        connected_featurizer.get_node_features(lig, source="ligand", complex_bool=True)
    )
    complex_features = np.concatenate(
        (protein_features_complex, ligand_features_complex), axis=0
    )
    complex_coords = np.concatenate(
        (protein_coords_complex, ligand_coords_complex), axis=0
    )
    complex_edges, complex_attrs = (
        connected_featurizer.get_protein_ligand_complex_edges(prot, lig)
    )

    # Process unconnected protein
    full_prot_withH_features, prot_withH_coords = (
        unconnected_featurizer.get_node_features(
            full_prot_withH, source="protein", complex_bool=False
        )
    )
    full_prot_withH_coords, full_prot_withH_atom_types, full_prot_withH_features = (
        np.array(prot_withH_coords),
        np.array(full_prot_withH_features[:, :12]),
        np.array(full_prot_withH_features[:, 12:]),
    )

    return (
        protein_coords,
        protein_features,
        np.array(prot_edges),
        np.array(prot_attrs),
        ligand_coords,
        ligand_features,
        np.array(lig_edges),
        np.array(lig_attrs),
        complex_coords,
        complex_features,
        complex_edges,
        complex_attrs,
        full_prot_withH_coords,
        full_prot_withH_atom_types,
        full_prot_withH_features,
    )


def _process_surface_features(
    data: dict[str, torch.Tensor], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    logger.info("Processing surface features")
    torch.manual_seed(42)

    P = {}
    L = {}

    # Atom information:
    P["atoms"] = data["atom_coords"].to(device)
    P["batch_atoms"] = data["atom_coords_batch"].to(device)
    # Chemical features: atom coordinates and types.
    P["atom_xyz"] = data["atom_coords"].to(device)
    P["atomtypes"] = data["atom_types"].to(device)
    P["atom_features"] = data["atom_features"].to(device)

    L["xyz"] = data["ligand_coords"].to(device)
    L["batch"] = data["ligand_coords_batch"].to(device)

    P["xyz"], P["normals"], P["batch"] = _atoms_to_points_normals(
        atoms=P["atoms"].to(device),
        batch=P["batch_atoms"].to(device),
        atomtypes=P["atomtypes"].to(device),
    )
    P["xyz"], P["normals"], P["batch"] = _select_surface_pocket(P, L)
    return P["xyz"], P["normals"]


# need to add better comments here, todo
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
            atoms, z, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes
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


def _scale_graph_features(
    ligand_node_features: np.ndarray,
    ligand_edge_attrs: np.ndarray,
    protein_node_features: np.ndarray,
    protein_edge_attrs: np.ndarray,
    complex_node_features: np.ndarray,
    complex_edge_attrs: np.ndarray,
    scaler_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale the continuous graph features for a single datapoint using pre-fitted scalers.

    Args:
        ligand_node_features: Numpy array of ligand node features.
        ligand_edge_attrs: Numpy array of ligand edge attributes.
        protein_node_features: Numpy array of protein node features.
        protein_edge_attrs: Numpy array of protein edge attributes.
        complex_node_features: Numpy array of complex node features.
        complex_edge_attrs: Numpy array of complex edge attributes.
        scaler_file: Path to the pickle file containing pre-fitted scalers
                     (a dict with 'node_scaler' and 'edge_scaler').

    Returns:
        tuple: Scaled versions of:
               (ligand_node_features_scaled,
                ligand_edge_attrs_scaled,
                protein_node_features_scaled,
                protein_edge_attrs_scaled,
                complex_node_features_scaled,
                complex_edge_attrs_scaled)

    """
    logger.info("Scaling graph features")

    # Continuous feature indices for node and edge features
    node_continuous_indices = [-4, -3, -2, -1]  # Last 4 indices for node features
    edge_continuous_indices = [1, 4, 5]  # Indices 1, 4, and 5 for edge features

    # Load the pre-fitted scalers
    with open(scaler_file, "rb") as f:
        scalers = pickle.load(f)
        node_scaler = scalers["node_scaler"]
        edge_scaler = scalers["edge_scaler"]

    def standardize_continuous_features(
        features: np.ndarray, continuous_indices: list[int], scaler: StandardScaler
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

    # Scale ligand node features
    ligand_node_features_scaled = standardize_continuous_features(
        ligand_node_features, node_continuous_indices, node_scaler
    )

    # Scale ligand edge attributes
    ligand_edge_attrs_scaled = standardize_continuous_features(
        ligand_edge_attrs, edge_continuous_indices, edge_scaler
    )

    # Scale protein node features
    protein_node_features_scaled = standardize_continuous_features(
        protein_node_features, node_continuous_indices, node_scaler
    )

    # Scale protein edge attributes
    protein_edge_attrs_scaled = standardize_continuous_features(
        protein_edge_attrs, edge_continuous_indices, edge_scaler
    )

    # Scale complex node features
    complex_node_features_scaled = standardize_continuous_features(
        complex_node_features, node_continuous_indices, node_scaler
    )

    # Scale complex edge attributes
    complex_edge_attrs_scaled = standardize_continuous_features(
        complex_edge_attrs, edge_continuous_indices, edge_scaler
    )

    return (
        ligand_node_features_scaled,
        ligand_edge_attrs_scaled,
        protein_node_features_scaled,
        protein_edge_attrs_scaled,
        complex_node_features_scaled,
        complex_edge_attrs_scaled,
    )


def _scale_transformer_embedding(
    transformer_embedding: np.ndarray, scaler_file: Path
) -> np.ndarray:
    """
    Given a transformer-based feature embedding and a pre-trained scaler file,
    this function returns the standardized (scaled) version of that embedding.

    Args:
        transformer_embedding: The original transformer feature embedding to be scaled.
                                         Should be 1D or 2D. If 1D, it will be reshaped.
        scaler_file: Path to the pickle file containing the pre-trained scaler
                           dictionary with a key 'roberta_scaler'.

    Returns:
        The scaled transformer feature embedding as a 1D NumPy array.
    """
    logger.info("Scaling transformer embedding")
    # Load the pre-trained scaler
    with open(scaler_file, "rb") as f:
        roberta_scaler = pickle.load(f)["roberta_scaler"]

    # Ensure the embedding is 2D for the scaler
    if transformer_embedding.ndim == 1:
        transformer_embedding = transformer_embedding.reshape(1, -1)

    # Scale the embedding
    standardized_embedding = roberta_scaler.transform(transformer_embedding)

    # Squeeze back to 1D array if applicable
    return standardized_embedding.squeeze()


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
    with open(scaler_file, "rb") as f:
        rdkit_scaler = pickle.load(f)["rdkit_scaler"]

    # Ensure the vector is 2D for the scaler
    if rdkit_vector.ndim == 1:
        rdkit_vector = rdkit_vector.reshape(1, -1)

    # Scale the vector
    standardized_vector = rdkit_scaler.transform(rdkit_vector)

    # Handle NaN values by replacing them with the corresponding mean
    if np.isnan(standardized_vector).any():
        feature_means = rdkit_scaler.mean_  # Means of the features from the scaler
        standardized_vector = np.where(
            np.isnan(standardized_vector), feature_means, standardized_vector
        )

    # Return as a 1D array if it was originally 1D
    if standardized_vector.shape[0] == 1:
        return standardized_vector.squeeze()
    else:
        return standardized_vector


def _scale_esm2_embedding(esm2_embedding: np.ndarray, scaler_file: Path) -> np.ndarray:
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
    with open(scaler_file, "rb") as f:
        esm2_scaler = pickle.load(f)["esm2_scaler"]

    # Ensure the embedding is 2D for the scaler
    if esm2_embedding.ndim == 1:
        esm2_embedding = esm2_embedding.reshape(1, -1)

    # Scale the embedding
    standardized_embedding = esm2_scaler.transform(esm2_embedding)

    # Return as a 1D array if it was originally 1D
    if standardized_embedding.shape[0] == 1:
        return standardized_embedding.squeeze()
    else:
        return standardized_embedding


def _scale_unconnected_graph_features(
    protein_node_features: np.ndarray, scaler_file: Path
) -> np.ndarray:
    """
    Given protein node features and a pre-trained scaler file, this function standardizes
    the continuous features and returns the standardized features.

    Args:
        protein_node_features: The original protein node features array.
                                            Should be 2D (num_nodes x num_features).
        scaler_file: Path to the pickle file containing the pre-trained scaler
                           dictionary with a key 'protein_node_scaler'.

    Returns:
        The standardized protein node features array.
    """
    logger.info("Scaling unconnected graph features")

    # Continuous feature indices (last 4 indices are continuous)
    continuous_indices = [-4, -3, -2, -1]

    # Load the pre-trained scaler
    with open(scaler_file, "rb") as f:
        protein_node_scaler = pickle.load(f)["protein_node_scaler"]

    # Ensure the input features are a NumPy array
    protein_node_features = np.asarray(protein_node_features)

    # Split continuous and non-continuous features
    continuous_features = protein_node_features[:, continuous_indices]
    non_continuous_features = np.delete(
        protein_node_features, continuous_indices, axis=1
    )

    # Standardize the continuous features
    standardized_continuous_features = protein_node_scaler.transform(
        continuous_features
    )

    # Concatenate back non-continuous and standardized continuous features
    standardized_features = np.concatenate(
        [non_continuous_features, standardized_continuous_features], axis=1
    )

    return standardized_features


def run_single_inference(
    ckpt_path: Path,
    scaled_esm2_embedding: np.ndarray,
    scaled_pocket_features: np.ndarray,
    pocket_coords: np.ndarray,
    pocket_edges: np.ndarray,
    scaled_pocket_edge_attrs: np.ndarray,
    protein_coords: np.ndarray,
    protein_atom_types: np.ndarray,
    scaled_protein_features: np.ndarray,
    protein_surface_coords: torch.Tensor | None,
    protein_surface_norms: torch.Tensor | None,
    scaled_rdkit_vector: np.ndarray,
    scaled_transformer_embedding: np.ndarray,
    scaled_ligand_features: np.ndarray,
    ligand_coords: np.ndarray,
    ligand_edges: np.ndarray,
    scaled_ligand_edge_attrs: np.ndarray,
    scaled_complex_features: np.ndarray,
    complex_coords: np.ndarray,
    complex_edges: np.ndarray,
    scaled_complex_edge_attrs: np.ndarray,
    calc_surface_features: bool,
    device: str,
) -> float:
    """
    Given all the processed feature arrays for a single protein-ligand complex,
    this function:
    1. Loads the T-ALPHA model from a checkpoint.
    2. Constructs a data dictionary as done in the inference code.
    3. Performs a forward pass to predict binding affinity.
    4. Returns the predicted binding affinity (float).

    Args:
        ckpt_path: Path to the model checkpoint (.ckpt) file.
        ... (All the scaled and processed inputs as numpy arrays or torch tensors)
        device: The computation device ('cuda' or 'cpu').

    Returns:
        The predicted binding affinity.
    """

    # Ensure tensors are on the correct device
    def to_tensor(x, dtype=torch.float32):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=dtype)
        return x.to(device)

    # Convert all inputs to torch tensors on the correct device
    esm_vector = to_tensor(scaled_esm2_embedding)  # shape [1, D]
    rdkit_vector = to_tensor(scaled_rdkit_vector)  # shape [1, D]
    roberta_vector = to_tensor(scaled_transformer_embedding)  # shape [1, D]
    atom_coords = to_tensor(protein_coords)  # shape [N_atoms, 3]
    atom_features = to_tensor(
        np.concatenate([protein_atom_types, scaled_protein_features], axis=1)
    )  # shape [N_atoms, D]

    if calc_surface_features:
        surface_coords = to_tensor(protein_surface_coords)  # shape [N_surf, 3]
        surface_normals = to_tensor(protein_surface_norms)  # shape [N_surf, 3]

        surface_batch_idx = torch.zeros(
            surface_coords.size(0), dtype=torch.long, device=device
        )
    else:
        surface_coords = None
        surface_normals = None
        surface_batch_idx = None

    # For a single sample, all batch indices are 0
    atom_coords_batch = torch.zeros(
        atom_coords.size(0), dtype=torch.long, device=device
    )

    # Protein graph
    logger.info("Generating protein graph dataset")
    protein_graph_data = Data(
        node_feats=to_tensor(scaled_pocket_features),
        node_coords=to_tensor(pocket_coords),
        edge_index=to_tensor(pocket_edges, dtype=torch.long).t().contiguous(),
        edge_attr=to_tensor(scaled_pocket_edge_attrs),
    )
    protein_graph = Batch.from_data_list([protein_graph_data]).to(device)  # type: ignore
    protein_graph_batch = torch.zeros(
        protein_graph_data.node_feats.size(0), dtype=torch.long, device=device
    )

    # Ligand graph
    logger.info("Generating ligand graph dataset")
    ligand_graph_data = Data(
        node_feats=to_tensor(scaled_ligand_features),
        node_coords=to_tensor(ligand_coords),
        edge_index=to_tensor(ligand_edges, dtype=torch.long).t().contiguous(),
        edge_attr=to_tensor(scaled_ligand_edge_attrs),
    )
    ligand_graph = Batch.from_data_list([ligand_graph_data]).to(  # type: ignore
        device
    )  # typ  e: ignore
    ligand_graph_batch = torch.zeros(
        ligand_graph_data.node_feats.size(0), dtype=torch.long, device=device
    )

    # Complex graph
    logger.info("Generating complex graph dataset")
    complex_graph_data = Data(
        node_feats=to_tensor(scaled_complex_features),
        node_coords=to_tensor(complex_coords),
        edge_index=to_tensor(complex_edges, dtype=torch.long).t().contiguous(),
        edge_attr=to_tensor(scaled_complex_edge_attrs),
    )
    complex_graph = Batch.from_data_list([complex_graph_data]).to(device)  # type: ignore
    complex_graph_batch = torch.zeros(
        complex_graph_data.node_feats.size(0), dtype=torch.long, device=device
    )

    # Construct the data dictionary
    # Operator, label, pdbid are dummy since we only need a prediction
    data = {
        "esm_vector": esm_vector.unsqueeze(0),  # Ensure [1, D]
        "rdkit_vector": rdkit_vector.unsqueeze(0),
        "roberta_vector": roberta_vector.unsqueeze(0),
        "atom_coords_batch": atom_coords_batch,
        "atom_coords": atom_coords,
        "atom_features": atom_features,
        "surface_coords": surface_coords,
        "surface_normals": surface_normals,
        "surface_batch_idx": surface_batch_idx,
        "protein_graph": protein_graph,
        "protein_graph_batch": protein_graph_batch,
        "ligand_graph": ligand_graph,
        "ligand_graph_batch": ligand_graph_batch,
        "complex_graph": complex_graph,
        "complex_graph_batch": complex_graph_batch,
    }

    # Load the model
    logger.info("Loading model from checkpoint")
    model = MetaModel(
        device=device,
        use_protein_graph=True,
        use_protein_surface=calc_surface_features,
        use_protein_sequence=True,
        use_ligand_properties=True,
        use_ligand_graph=True,
        use_ligand_sequence=True,
        use_complex_graph=True,
    )
    lightning_model = MetaModelLightning.load_from_checkpoint(
        train_dataset=None,
        val_dataset=None,
        checkpoint_path=ckpt_path,
        model=model,
        batch_size=1,
        num_epochs=0,
        batch_norm=True,
    ).to(device)
    lightning_model.eval()

    # Forward pass
    logger.info("Performing forward pass")
    with torch.no_grad():
        output = lightning_model.model(data)  # output is a tensor of shape [1, 1]

    # output is the predicted binding affinity
    return output.cpu().numpy().flatten()[0]


def run(
    protein_file: Path,
    ligand_file: Path,
    esm_model_name: ESMModel,
    t_alpha_model_parameters_file: Path = Path("T-ALPHA_params.ckpt"),
    smiles_transformer_model_parameters_file: Path = Path(
        "SMILES_transformer_params.pt"
    ),
    smiles_transformer_training_data_file: Path = Path(
        "SMILES_transformer_pretraining_data.parquet"
    ),
    connected_graph_scaler_name: str = "connected_graph_scaler.pkl",
    unconnected_graph_scaler_name: str = "unconnected_graph_scaler.pkl",
    ligand_sequence_scaler_name: str = "ligand_sequence_scaler.pkl",
    protein_sequence_scaler_name: str = "protein_sequence_scaler.pkl",
    ligand_properties_scaler_name: str = "ligand_properties_scaler.pkl",
    smiles_transformer_vocab_name: str = "smiles_transformer_vocab.json.zst",
    calc_surface_features: bool = True,
    device: str | None = None,
):
    """
    T-ALPHA: Protein-Ligand Binding Affinity Prediction Pipeline

    This function implements the T-ALPHA pipeline for predicting the binding affinity (pKd) of protein-ligand complexes.
    The pipeline supports multiple input modes:
    1. Provide protein and ligand files
    2. Provide protein amino acid sequence and ligand SMILES string.
    3. Provide a PDB ID to fetch the protein sequence and ligand SMILES string.

    Features:
    - Predicts 3D structures from sequences using the CHAI1 module.
    - Extracts ESM2 embeddings, RDKit 2D descriptors, and SMILES transformer embeddings.
    - Constructs protein and ligand graphs with detailed atomic and bond features.
    - Generates a surface-oriented point cloud for the protein.
    - Scales and preprocesses all features to match the trained T-ALPHA model.
    - Runs inference using the trained T-ALPHA model to predict binding affinity.
    - Visualizes the protein-ligand complex with Py3Dmol.

    Output:
    - Predicted binding affinity (pKd) of the protein-ligand complex.
    - Optional 3D visualization of the protein-ligand complex.

    Parameters:
        protein_file: Path to the protein file.
        ligand_file: Path to the ligand file.
        esm_model_name: ESM model name
        t_alpha_model_parameters_file: Path to the model parameters file.
        smiles_transformer_model_parameters_file: Path to the model parameters file.
        smiles_transformer_training_data_file: Path to the training data file.
        connected_graph_scaler_name: Path to the connected graph scaler file.
        unconnected_graph_scaler_name: Path to the unconnected graph scaler file.
        ligand_sequence_scaler_name: Path to the ligand sequence scaler file.
        protein_sequence_scaler_name: Path to the protein sequence scaler file.
        ligand_properties_scaler_name: Path to the ligand properties scaler file.
        calc_surface_features: Whether to calculate surface features.
        device: The device to use for computation.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    smiles_transformer_vocab = create_or_load_smiles_transformer_vocab(
        smiles_transformer_training_data_file, smiles_transformer_vocab_name
    )

    logger.info("Loading transformer feature extractor")
    transformer_feature_extractor = TransformerFeatureExtractor(
        model_parameters_file=smiles_transformer_model_parameters_file,
        smiles_transformer_vocab=smiles_transformer_vocab,
        device=device,
    )

    # Add hydrogens to input files if provided
    logger.info("Preparing protein and ligand files...")

    # Add hydrogens to the protein
    protein_molecule = next(pybel.readfile("pdb", str(protein_file)))
    _add_hydrogens_and_save(protein_molecule, protein_file, "pdb")

    # Add hydrogens to the ligand
    ligand_format = ligand_file.suffix.lstrip(".")
    ligand_molecule = next(pybel.readfile(ligand_format, str(ligand_file)))
    _add_hydrogens_and_save(ligand_molecule, ligand_file, ligand_format)

    logger.info("Featurizing data...")

    # Extract ESM2 embedding
    protein_seq = _protein_mol_to_seq(protein_molecule)
    esm2_embedding = _get_esm2_embedding(protein_seq, esm_model_name)

    # Extract RDKit 2D descriptor vector
    rdkit_mol = _pybel_mol_to_rdkit_mol(ligand_molecule)
    rdkit_vector = _get_rdkit_vector(rdkit_mol)

    # Extract SMILES transformer encoder embedding
    transformer_vector = _get_transformer_vector(
        rdkit_mol, transformer_feature_extractor
    )

    # Extract protein pocket
    protein_pocket_molecule = _extract_protein_pocket(protein_molecule, ligand_molecule)

    # Define graph featurizer variables
    connected_featurizer = GraphFeaturizer()
    unconnected_featurizer = GraphFeaturizer(surface_features_bool=True)

    # Obtain graphs
    (
        pocket_coords,
        pocket_features,
        pocket_edges,
        pocket_edge_attrs,
        ligand_coords,
        ligand_features,
        ligand_edges,
        ligand_edge_attrs,
        complex_coords,
        complex_features,
        complex_edges,
        complex_edge_attrs,
        protein_coords,
        protein_atom_types,
        protein_features,
    ) = _get_graphs(
        connected_featurizer=connected_featurizer,
        unconnected_featurizer=unconnected_featurizer,
        prot=protein_pocket_molecule,
        lig=ligand_molecule,
        full_prot_withH=protein_molecule,
    )

    # Define data objects needed to obtain surface-oriented point cloud
    data = {
        "atom_coords": torch.tensor(protein_coords),
        "atom_coords_batch": torch.zeros(protein_coords.shape[0], dtype=torch.long),
        "atom_types": torch.tensor(protein_atom_types),
        "atom_features": torch.tensor(protein_features),
        "ligand_coords": torch.tensor(ligand_coords),
        "ligand_coords_batch": torch.zeros(ligand_coords.shape[0], dtype=torch.long),
    }

    # Obtain surface coordinates and normals
    if calc_surface_features:
        protein_surface_coords_tensor, protein_surface_norms_tensor = (
            _process_surface_features(data, device=device)
        )
    else:
        protein_surface_coords_tensor = None
        protein_surface_norms_tensor = None

    # Scale the featurized connected graphs
    with importlib.resources.path(
        RESOURCES_BASE, connected_graph_scaler_name
    ) as connected_graph_scaler_file:
        (
            scaled_ligand_features,
            scaled_ligand_edge_attrs,
            scaled_pocket_features,
            scaled_pocket_edge_attrs,
            scaled_complex_features,
            scaled_complex_edge_attrs,
        ) = _scale_graph_features(
            ligand_node_features=ligand_features,
            ligand_edge_attrs=ligand_edge_attrs,
            protein_node_features=pocket_features,
            protein_edge_attrs=pocket_edge_attrs,
            complex_node_features=complex_features,
            complex_edge_attrs=complex_edge_attrs,
            scaler_file=connected_graph_scaler_file,
        )

    # Scale the SMILES transformer encoder embedding
    with importlib.resources.path(
        RESOURCES_BASE, ligand_sequence_scaler_name
    ) as ligand_sequence_scaler_file:
        scaled_transformer_embedding = _scale_transformer_embedding(
            transformer_embedding=transformer_vector,
            scaler_file=ligand_sequence_scaler_file,
        )

    # Scale the RDKit 2D descriptor vector
    with importlib.resources.path(
        RESOURCES_BASE, ligand_properties_scaler_name
    ) as ligand_properties_scaler_file:
        scaled_rdkit_vector = _scale_rdkit_vector(
            rdkit_vector=rdkit_vector,
            scaler_file=ligand_properties_scaler_file,
        )

    # Scale the ESM2 embedding
    with importlib.resources.path(
        RESOURCES_BASE, protein_sequence_scaler_name
    ) as protein_sequence_scaler_file:
        scaled_esm2_embedding = _scale_esm2_embedding(
            esm2_embedding=esm2_embedding,
            scaler_file=protein_sequence_scaler_file,
        )

    # Scale the featurized unconnected graphs
    with importlib.resources.path(
        RESOURCES_BASE, unconnected_graph_scaler_name
    ) as unconnected_graph_scaler_file:
        scaled_protein_features = _scale_unconnected_graph_features(
            protein_node_features=protein_features,
            scaler_file=unconnected_graph_scaler_file,
        )

    # Update keys in parameter file
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_t_alpha_model_parameters_file = (
            Path(temp_dir) / "updated_T-ALPHA_params.ckpt"
        )
        update_parameter_keys(
            t_alpha_model_parameters_file, tmp_t_alpha_model_parameters_file
        )

        logger.info("Performing T-ALPHA inference...")

        prediction = run_single_inference(
            ckpt_path=tmp_t_alpha_model_parameters_file,
            scaled_esm2_embedding=scaled_esm2_embedding,
            scaled_pocket_features=scaled_pocket_features,
            pocket_coords=pocket_coords,
            pocket_edges=pocket_edges,
            scaled_pocket_edge_attrs=scaled_pocket_edge_attrs,
            protein_coords=protein_coords,
            protein_atom_types=protein_atom_types,
            scaled_protein_features=scaled_protein_features,
            protein_surface_coords=protein_surface_coords_tensor,
            protein_surface_norms=protein_surface_norms_tensor,
            scaled_rdkit_vector=scaled_rdkit_vector,
            scaled_transformer_embedding=scaled_transformer_embedding,
            scaled_ligand_features=scaled_ligand_features,
            ligand_coords=ligand_coords,
            ligand_edges=ligand_edges,
            scaled_ligand_edge_attrs=scaled_ligand_edge_attrs,
            scaled_complex_features=scaled_complex_features,
            complex_coords=complex_coords,
            complex_edges=complex_edges,
            scaled_complex_edge_attrs=scaled_complex_edge_attrs,
            calc_surface_features=calc_surface_features,
            device=device,
        )
        logger.info(f"Prediction: {prediction}")

        logger.info("Successfully completed T-ALPHA inference.")


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
                smiles_list = pd.read_parquet(smiles_transformer_training_data_file)[
                    "SMILES"
                ].tolist()

                smiles_transformer_vocab = SMILESTransformerVocab.from_smiles(
                    smiles_list
                )
                f.write(smiles_transformer_vocab.model_dump_json())

    return smiles_transformer_vocab
