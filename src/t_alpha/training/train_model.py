import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from t_alpha.data.full_model_dataset import MetaModelDataset
from t_alpha.models.full_model import MetaModel
from t_alpha.training.lightning_module import MetaModelLightning


def train_model(
    device: str,
    train_set: str,
    val_set: str,
    batch_size: int = 32,
    num_epochs: int = 120,
    batch_norm: bool = True,
    patience: int = 100,
    n_epochs: int = 120,
    checkpoint_dir: str | None = None,
    save_dir: str | None = None,
    save_name: str | None = None,
    model_checkpoint_path: str | None = None,
    use_protein_graph: bool = True,
    use_protein_surface: bool = True,
    use_protein_sequence: bool = True,
    use_ligand_properties: bool = True,
    use_ligand_graph: bool = True,
    use_ligand_sequence: bool = True,
    use_complex_graph: bool = True,
):
    """ "
    Trains the MetaModel using PyTorch Lightning with checkpointing and logging.

    Args:
        device: Device for training ('cuda' or 'cpu'). Defaults to 'cuda'.
        train_set: Path to the HDF5 training dataset.
        val_set: Path to the HDF5 validation dataset.
        batch_size: Batch size for training and validation. Defaults to 32.
        num_epochs: Total number of training epochs. Defaults to 120.
        batch_norm: Whether to use batch normalization. Defaults to True.
        patience: Early stopping patience. Defaults to 100.
        n_epochs: Number of epochs for training. Defaults to 120.
        checkpoint_dir: Directory to save model checkpoints.
        save_dir: Directory to save the experiment logs.
        save_name: Name for the experiment logs.
        model_checkpoint_path: Path to a pre-trained model checkpoint. Defaults to None.
        optimizer_checkpoint_path: Path to an optimizer checkpoint. Defaults to None.
    """

    # Initialize the model (do not move it to device yet)
    model = MetaModel(
        device=device,
        use_protein_graph=use_protein_graph,
        use_protein_surface=use_protein_surface,
        use_protein_sequence=use_protein_sequence,
        use_ligand_properties=use_ligand_properties,
        use_ligand_graph=use_ligand_graph,
        use_ligand_sequence=use_ligand_sequence,
        use_complex_graph=use_complex_graph,
        batch_norm=batch_norm,
    ).to(device)

    train_dataset = MetaModelDataset(train_set, device)
    val_dataset = MetaModelDataset(val_set, device)

    lightning_model = MetaModelLightning(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        batch_norm=batch_norm,
        patience=patience,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model-{epoch:02d}",
        save_top_k=1,
        mode="max",
        monitor="val_correlation",
        verbose=True,
    )

    # Create a CSVLogger with custom options
    logger = CSVLogger(
        save_dir=save_dir,
        name=save_name,
    )

    trainer = pl.Trainer(
        devices=4,
        strategy="fsdp",
        accelerator="cuda",
        max_epochs=n_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=0.1,
        gradient_clip_algorithm="value",
    )

    if model_checkpoint_path is not None:
        print(f"Loading model from {model_checkpoint_path}")
        trainer.fit(lightning_model, ckpt_path=model_checkpoint_path)

    else:
        trainer.fit(lightning_model)
