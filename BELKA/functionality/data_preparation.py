from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import lightgbm as lgb
from lightgbm import LGBMClassifier
from pytorch_lightning.callbacks import ModelCheckpoint
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel
import numpy as np
import tqdm
import os 
from torch.utils.data import IterableDataset
import h5py
import pickle
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class IterableCNNDataset(IterableDataset):
    def __init__(self, encoded_path, batch_size=1000):
        """
        Args:
            encoded_path (str): Path to the Parquet file containing embeddings and labels
            batch_size (int): Number of rows to read at once for efficiency
        """
        self.encoded_path = encoded_path
        self.batch_size = batch_size

        self.length = self._get_length()

    def _get_length(self):
        df = pd.read_parquet(self.embeddings_path, columns=["label"])
        return len(df)

    def __len__(self):
        return self.length

    def __iter__(self):
        """Yields embeddings and labels one-by-one from the Parquet file"""
        df_iter = pd.read_parquet(self.encoded_path, iterator=True, chunksize=self.batch_size)
        
        for df_batch in df_iter:
            labels = df_batch["label"].values
            encoded_smiles = df_batch.drop(columns=["label"]).values 
            
            for emb, lbl in zip(encoded_smiles, labels):
                yield torch.tensor(emb, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)


class IterableEmbDataset(IterableDataset):
    def __init__(self, embeddings_path, batch_size=1000):
        """
        Args:
            embeddings_path (str): Path to the Parquet file containing embeddings and labels
            batch_size (int): Number of rows to read at once for efficiency
        """
        self.embeddings_path = embeddings_path
        self.batch_size = batch_size

        self.length = self._get_length()

    def _get_length(self):
        df = pd.read_parquet(self.embeddings_path, columns=["label"])
        return len(df)

    def __len__(self):
        return self.length

    def __iter__(self):
        """Yields embeddings and labels one-by-one from the Parquet file"""
        df_iter = pd.read_parquet(self.embeddings_path, iterator=True, chunksize=self.batch_size)
        
        for df_batch in df_iter:
            labels = df_batch["label"].values
            embeddings = df_batch.drop(columns=["label"]).values 
            
            for emb, lbl in zip(embeddings, labels):
                yield torch.tensor(emb, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)



def train_model(model_name, protein_name, model, train_loader, val_loader, epochs=20):
    """
    Trainer and checkpoint initialization
    
    Args:
        model_name (str): The name of the model (for checkpointing)
        protein_name (str): The name of the protein (for checkpointing)
        model: The model instance (e.g., CNN, ChemBERTa, etc.)
        train_loader: The training data loader
        val_loader: The validation data loader
        epochs (int): Number of training epochs

    Returns:
        trainer: The trainer object after training
    """
    
    logger = CSVLogger("logs", name=model_name)

    # Define checkpointing: Save best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="../checkpoints",  
        filename=f"{protein_name}_{model_name}-{{epoch}}-{{val_loss:.4f}}", 
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=2,
        #callbacks=[checkpoint_callback],
        logger = logger
    )
    trainer.fit(model, train_loader, val_loader)

    return trainer


def train_gbt(protein_name, train_loader, lgb_params=None, save_dir="./checkpoints"):
    """
    Function to train the LightGBM model in batches
    
    Args:
        protein_name (str): The name of the protein (for saving model)
        train_loader: The training data loader
        lgb_params (dict): Parameters for the LightGBM classifier
        save_dir (str): Directory to save the trained model

    Returns:
        lgb_cls: The trained LightGBM model
    """

    lgb_cls = LGBMClassifier(**lgb_params)

    for features_batch, labels_batch in tqdm(train_loader, desc=f"Training {protein_name} (Training)"):
        
        X_batch = features_batch.numpy() 
        y_batch = labels_batch.numpy()
        
        lgb_cls.fit(X_batch, y_batch, eval_metric='auc', init_model=lgb_cls.booster_ if hasattr(lgb_cls, "booster_") else None)
    
    lgb_cls.save_model(f"{save_dir}/{protein_name}_lgb_model.txt")

    return lgb_cls




