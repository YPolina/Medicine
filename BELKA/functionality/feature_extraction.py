import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from transformers import AutoTokenizer, AutoModel

protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
#protein_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")


class FeatureEngineering:

    def __init__(self, batch, feature_type, proteins=['HSA', 'BRD4', 'sEH'], 
                 model=None, tokenizer=None, max_length=142):
        """

        Args:
            batch (pd.DataFrame): Batch of data
            feature_type (str): 'GBT', 'CNN', or 'embeddings'
            proteins (list): List of protein names for consistent one-hot encoding
            model (transformers.PreTrainedModel, optional): Transformer model for SMILES embeddings
            tokenizer (transformers.PreTrainedTokenizer, optional): Tokenizer for SMILES
            max_length (int, optional): Max length for CNN SMILES encoding

        """
        self.batch = batch.copy()
        self.feature_type = feature_type
        self.proteins = proteins
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def replace_smiles(self, smile):

        """
        Replaces [Dy] with H in SMILES

        """
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return smile 
        
        m_Dy = Chem.MolFromSmiles("[Dy]")
        m_H = Chem.MolFromSmiles("H") 

        mol = AllChem.ReplaceSubstructs(mol, m_Dy, m_H)[0]
        return Chem.MolToSmiles(mol)

    def compute_features(self):
        """
        Calls the appropriate feature extraction method and includes protein encoding

        """
        self.batch["molecule_smiles"] = self.batch["molecule_smiles"].apply(self.replace_smiles)
        
        if self.feature_type == "GBT":
            X_batch = self.GBT_features()
        elif self.feature_type == "CNN":
            X_batch = self.CNN_features()
        elif self.feature_type == "embeddings":
            X_batch = self.compute_embeddings()
        else:
            raise ValueError("Invalid feature_type. Choose from 'GBT', 'CNN', or 'embeddings'.")


        return X_batch

    def GBT_features(self):
        """
        Computes features for tree-based modelsfingerprints for specified SMILES columns

        Returns:
            X: pd.DataFrame - features for batch: fingerprints and encoded proteins
        """
        #FingerprintFenerator initialization
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, useBondTypes=True, includeChirality=True, includeRingMembership=True)

        def compute_fn(smile):
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                return np.zeros(1024, dtype=np.int8)
            fp = fpg.GetFingerprint(mol)
            arr = np.zeros(1024, dtype=np.int8)
            ConvertToNumpyArray(fp, arr)
            return arr

        self.batch['fp'] = self.batch['molecule_smiles'].apply(compute_fn)
        fingerprint_df = pd.DataFrame(self.batch["fp"].to_list(), index=self.batch.index)

        return fingerprint_df

    def CNN_features(self):
        """
        Computes features for CNN models


        Returns:
            X: pd.DataFrame - features for batch: encoded smiles strings
        """
        enc_dict = {
            'l': 1, 'y': 2, '@': 3, '3': 4, 'H': 5, 'S': 6, 'F': 7, 'C': 8, 'r': 9, 's': 10, '/': 11, 'c': 12, 'o': 13,
            '+': 14, 'I': 15, '5': 16, '(': 17, '2': 18, ')': 19, '9': 20, 'i': 21, '#': 22, '6': 23, '8': 24, '4': 25, '=': 26,
            '1': 27, 'O': 28, '[': 29, 'D': 30, 'B': 31, ']': 32, 'N': 33, '7': 34, 'n': 35, '-': 36
        }
        
        def encode_smile(smile):
            encoded = [enc_dict.get(char, 0) for char in smile]  
            padded = encoded + [0] * (self.max_length - len(encoded))
            return np.array(padded[:self.max_length], dtype=np.uint8)

        smiles_enc = np.stack([encode_smile(smile) for smile in self.batch['molecule_smiles']])
        feature_df = pd.DataFrame(smiles_enc, columns=[f'enc_{i}' for i in range(self.max_length)])

        return feature_df

    def compute_embeddings(self):
        """
        Computes transformer embeddings for SMILES representations

        Returns:
            X: pd.DataFrame - features for batch: embeddings from smiles strings
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be provided for embeddings")

        smiles = self.batch['molecule_smiles']
        tokens = self.tokenizer(smiles, padding=True, truncation=True, max_length=512, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**tokens)

        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return pd.DataFrame(embeddings)

