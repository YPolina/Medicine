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
import numpy as np
import tqdm
import os 
import pickle


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, data):
        """
        Simple Dataset for SMILES and binary labels

        Args:
            data (pd.DataFrame): Data with 'molecule_smiles' and 'binds' columns
        """
        self.smiles = data['molecule_smiles'].values
        self.labels = data['binds'].values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]  
        label = torch.tensor(self.labels[idx], dtype=torch.float16) 
        return smile, label

def prepare_dataloaders(data, batch_size=100000):
    """
    Splits data into train/val sets and returns DataLoaders

    Args:
        data (pd.DataFrame): Data containing 'molecule_smiles' and 'binds'
        batch_size (int): Batch size

    Returns:
        train_loader, val_loader: PyTorch DataLoaders for training and validation
    """
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



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
    
    # Define checkpointing: Save best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",  
        filename=f"{protein_name}_{model_name}-{{epoch}}-{{val_loss:.4f}}", 
        monitor="val_loss", 
        save_top_k=3,
        mode="min",
        save_last=True,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=2,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader)

    return trainer

fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)

def compute_fps(smiles_batch):

    """
    Compute fingerprints for a batch of SMILES
    """
    results = []
    for smiles in smiles_batch:
        mol = Chem.MolFromSmiles(smiles.replace('[Dy]', '[H]'))
        if mol is None:
            results.append((np.zeros(1024, dtype=np.int8), {}))
        else:
            fp = fpg.GetCountFingerprint(mol)
            arr = np.zeros(1024, dtype=np.int8)
            ConvertToNumpyArray(fp, arr)
            results.append(arr)
    return np.array(results)

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

    # Iterate over the training data batches
    for smiles_batch, label_batch in tqdm(train_loader, desc=f"Training {protein_name} (Training)"):
        smiles_batch = [item['molecule_smiles'] for item in smiles_batch]
        labels_batch = [item['binds'] for item in label_batch]
        
        X_batch = compute_fps(smiles_batch) 
        y_batch = np.array(labels_batch)
        
        # Fit the model with the current batch and the full validation set
        lgb_cls.fit(X_batch, y_batch, eval_metric='auc', init_model=lgb_cls.booster_ if hasattr(lgb_cls, "booster_") else None)


    return lgb_cls




