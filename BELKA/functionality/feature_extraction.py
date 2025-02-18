from rdkit.Chem import rdFingerprintGenerator, DataStructs
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import torch


class GBT_features:

    '''
    Features for tree-based models

    '''
    def __init__(self, data):
        self.data = data.copy()
        self.fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, useBondTypes=True, includeChirality=True, includeRingMembership=True)

    def compute_fn(self, smiles):

        """
        Computes fingerprints for specified SMILES columns

        Args:
            
            smiles: SMILES strings

        Returns:
            arr: featureprints
            bit_info: information encoded in fingerprints
        """

        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(1024, dtype=np.int8)
        
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        
        
        fp = self.fpg.GetCountFingerprint(mol, additionalOutput=ao)

        bit_info = ao.GetBitInfoMap()
        arr = np.zeros(1024, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
    
        return arr, bit_info
    
    
    def transform(self, smiles_columns = ['molecule_smiles'], proteins = ['HSA', 'BRD4', 'sEH']):
        """
        Feature engineering based on fp and protein_name, concatenates them, and prepares feature matrix (X)

        Args:
            
            smiles_columns (list): List of column names containing SMILES strings
            proteins (list): List of all possible proteins across the whole data

        Returns:
            X (pd.DataFrame): Feature matrix containing concatenated fingerprints
        """

        results = self.data['molecule_smiles'].apply(self.compute_fn)
        self.data['fp'] = results.apply(lambda x: x[0])
        bit_infos = results.apply(lambda x: x[1]).tolist()

        protein_df = pd.get_dummies(self.data['protein_name'], prefix='protein', dtype = 'int')
        protein_df = protein_df.reindex(columns=[f'protein_{p}' for p in proteins], fill_value=0)

        fingerprint_df = pd.DataFrame(self.data["fp"].to_list(), index=self.data.index)

        self.data = pd.concat([self.data.drop(columns=["fp"]), fingerprint_df, protein_df], axis=1)

        columns = ['id', 'molecule_smiles', 'protein_name', 'buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'binds']

        features = [feature for feature in self.data.columns if feature not in columns]
        X = self.data[features]

        return X
    
class CNN_features:
    '''
    Features for CNN models

    '''

    def __init__(self, data, max_length=142):

        """
        data: data with SMILES columns 
        max_length: max length of the feature string
        enc_dict: encoding map, keys - SMILES symbols, values - corresponding number
        """
        
        self.data = data
        self.max_length = max_length
        self.enc_dict = {
            'l': 1, 'y': 2, '@': 3, '3': 4, 'H': 5, 'S': 6, 'F': 7, 'C': 8, 'r': 9, 's': 10, '/': 11, 'c': 12, 'o': 13,
            '+': 14, 'I': 15, '5': 16, '(': 17, '2': 18, ')': 19, '9': 20, 'i': 21, '#': 22, '6': 23, '8': 24, '4': 25, '=': 26,
            '1': 27, 'O': 28, '[': 29, 'D': 30, 'B': 31, ']': 32, 'N': 33, '7': 34, 'n': 35, '-': 36
        }
            
    def encode_smile(self, smile):
        """
        Encodes a single SMILES string into a numerical array

        Args:
            
            smiles: SMILES strings
        Returns:
            encoded smiles strings
        """
        encoded = [self.enc_dict.get(char, 0) for char in smile]  
        padded = encoded + [0] * (self.max_length - len(encoded))
        return np.array(padded, dtype=np.uint8)

    def transform(self, n_jobs=8):
        """
        Transforms the SMILES column into numerical features using parallel processing

        Args:
            
            n_jobs: Number of parallel jobs for encoding

        Returns:

            feature_df: DataFrame with encoded features

        """
        smiles = self.data['molecule_smiles'].values
        
     
        smiles_enc = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self.encode_smile)(smile) for smile in tqdm(smiles)
        )
        
        smiles_enc = np.stack(smiles_enc) 
        feature_df = pd.DataFrame(smiles_enc, columns=[f'enc_{i}' for i in range(self.max_length)])

        return feature_df
    

class Emb_features:

    def __init__(self, data):
        self.data = data
        self.model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    def smiles_embeddings(self, smiles:str):

        tokens = self.tokenizer(smiles, padding = True, truncation = True, max_length = 512, return_tensors = 'pt')
        
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        return outputs.last_hidden_state.mean(dim = 1).squeeze().cpu().numpy().tolist()

        
    
    

