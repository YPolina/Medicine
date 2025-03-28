import torch
import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator





class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for handling tokenized text data
    
    Args:
        tokenized_dataset (list or Dataset): A tokenized dataset containing dictionaries with:
                                                - 'input_ids' (torch.Tensor): The token IDs for the sample
                                                - 'attention_mask' (torch.Tensor): The attention mask for the sample
                                                - 'labels' (torch.Tensor, optional): The target labels if include_labels is True
            
        include_labels (bool): Whether to include labels in the returned items(true by default)
    """
    def __init__(self, tokenized_dataset, include_labels=True):
        """
        Initialize the CustomDataset
        
        Args:
            tokenized_dataset: The pre-tokenized dataset to wrap
            include_labels: Whether to include labels when returning items
        """
        self.tokenized_dataset = tokenized_dataset
        self.include_labels = include_labels

    def __len__(self):
        """
        Return the total number of samples in the dataset
        
        Returns:
            int: The length of the tokenized dataset
        """
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        """
        Retrieve and format a single sample from the dataset by index
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: A dictionary containing:
                - 'input_ids' (torch.Tensor): The token IDs for the sample
                - 'attention_mask' (torch.Tensor): The attention mask for the sample
                - 'labels' (torch.Tensor, optional): The target labels if include_labels is True
                
 
        """
        row = self.tokenized_dataset[idx]
        
        item = {
            'input_ids': torch.tensor(row['input_ids']),
            'attention_mask': torch.tensor(row['attention_mask']),
        }
        
        if self.include_labels:
            item['labels'] = torch.tensor(row['labels'])
        
        return item



class FingerprintDataset(Dataset):
    def __init__(self, smiles, labels=None, fp_size=700, bit_info=False):
        """
        Dataset that converts SMILES strings into Morgan Fingerprints

        Args:
            smiles (list or pd.Series): List of SMILES strings
            labels (list or pd.Series or None): List of binary labels
            fp_size (int): Size of the Morgan fingerprint vector
            bit_info (bool): Whether to compute additional bit information
        """
        self.smiles = smiles.tolist() if hasattr(smiles, "tolist") else smiles
        self.labels = labels if labels is not None else np.zeros(len(self.smiles), dtype=np.int64)
        self.fp_size = fp_size
        self.fpg = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=fp_size, includeChirality=True
        )
        self.bit_info = bit_info

    def __len__(self):
        """
        Return the total number of samples in the dataset
        
        Returns:
            int: Number of SMILES strings in the dataset
        """
        return len(self.smiles)

    def __getitem__(self, idx):

        """
        Get a single sample from the dataset by index
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: Contains three elements:
                - fingerprint (torch.Tensor): The Morgan fingerprint as float32 tensor
                - bit_info (dict): Dictionary containing bit information if requested
                - label (torch.Tensor): The corresponding label as float32 tensor
                
        Converts SMILES to molecular fingerprint and handles cases where SMILES parsing fails
        """
         
        smiles = self.smiles[idx]
        label = self.labels[idx]

        # Convert SMILES to Mol object
        mol = Chem.MolFromSmiles(smiles.replace("[Dy]", "[H]"))
        if mol is None:
            fingerprint = np.zeros(self.fp_size, dtype=np.int8)
            bit_info = {}
        else:
            if self.bit_info:
                ao = rdFingerprintGenerator.AdditionalOutput()
                ao.AllocateBitInfoMap()
                fp = self.fpg.GetCountFingerprint(mol, additionalOutput=ao)
                bit_info = ao.GetBitInfoMap()
            else:
                fp = self.fpg.GetFingerprint(mol)
                bit_info = {}

            fingerprint = np.zeros(self.fp_size, dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, fingerprint)

        return (
            torch.tensor(fingerprint, dtype=torch.float32),
            bit_info,
            torch.tensor(label, dtype=torch.float32),
        )

