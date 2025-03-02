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

class CustomDataset(Dataset):
    def __init__(self, data):
        """
        Simple Dataset for SMILES and binary labels

        Args:
            data (pd.DataFrame): Data with 'molecule_smiles' and 'binds' columns
        """
        self.labels = data['binds'].values

    def __len__(self):
        pass
    

    def __getitem__(self, idx):
        pass
    
class CNN_Dataset(CustomDataset):
    def __init__(self, data, max_length=142):
        """
        Dataset that generates additional features for SMILES strings

        Args:
            data (pd.DataFrame): Data with 'molecule_smiles' and 'binds' columns
            max_length (int): Maximum length for the SMILES encoding

        """
        super().__init__(data)
        self.max_length = max_length

    def create_features(self, smiles_batch):
        """
        Encodes SMILES strings into fixed-length numerical representations using CNN features.

        Args:
            smiles_batch (list of str): List of SMILES strings.

        Returns:
            Tensor: Encoded SMILES representations.
        """
        enc_dict = {
            'l': 1, 'y': 2, '@': 3, '3': 4, 'H': 5, 'S': 6, 'F': 7, 'C': 8, 'r': 9, 's': 10, '/': 11, 'c': 12, 'o': 13,
            '+': 14, 'I': 15, '5': 16, '(': 17, '2': 18, ')': 19, '9': 20, 'i': 21, '#': 22, '6': 23, '8': 24, '4': 25, '=': 26,
            '1': 27, 'O': 28, '[': 29, 'D': 30, 'B': 31, ']': 32, 'N': 33, '7': 34, 'n': 35, '-': 36
        }

        def encode_smile(smile):
            """Encodes a SMILES string to numerical values and pads it."""
            encoded = [enc_dict.get(char, 0) for char in smile]
            padded = encoded + [0] * (self.max_length - len(encoded))  # Padding to max_length
            return np.array(padded[:self.max_length], dtype=np.uint8)

        # Apply encoding to all SMILES in the batch
        smiles_enc = np.stack([encode_smile(smile) for smile in smiles_batch])
        return torch.tensor(smiles_enc, dtype=torch.float32)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.create_features(self.smiles[idx]), dtype = torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label
    

class IterableEmbDataset(IterableDataset):
    def __init__(self, embeddings_path):
        self.embeddings_path = embeddings_path
        with h5py.File(self.embeddings_path, "r") as h5f:
            self.length = len(h5f["embeddings"])

    def __len__(self):
        return self.length

    def __iter__(self):
        with h5py.File(self.embeddings_path, "r") as h5f:
            embeddings = h5f["embeddings"]
            labels = h5f["labels"]
            for i in range(len(embeddings)):
                yield torch.tensor(embeddings[i], dtype=torch.float32), torch.tensor(labels[i], dtype=torch.long)
    
class GBT_Dataset(CustomDataset):
    def __init__(self, data, ):
        
        super().__init__(data)

        self.fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)

    def create_features(self, smiles_batch):

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
    
    def __getitem__(self, idx):
       
        features = self.create_features([self.smiles[idx]]) 
        features = torch.tensor(features[0], dtype=torch.float32)  
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

class Llama_Dataset(CustomDataset):
    def __init__(self, data, model_name):
        """
        Dataset that generates additional features for SMILES strings

        Args:
            data (pd.DataFrame): Data with 'molecule_smiles' and 'binds' columns
            model_name (str): The pre-trained model name

        """
        super().__init__(data)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.model.eval()

    def create_features(self, smiles_batch):
        """
        Encodes SMILES strings into fixed-length numerical representations using CNN features.

        Args:
            smiles_batch (list of str): List of SMILES strings.

        Returns:
            Tensor: Tokenize SMILES representations.
        """
        tokens = self.tokenizer(smiles_batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        tokens = {key: val for key, val in tokens.items()}

        outputs = self.model(input_ids=tokens["input_ids"])
        pooled_output = outputs.last_hidden_state.mean(dim=1)

        return pooled_output

    def __getitem__(self, idx):
       
        # Create embeddings for the SMILES string
        #from [1, embedding_size] to [embedding_size])
        features = self.create_features([self.smiles[idx]]).squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label




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




