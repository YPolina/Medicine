import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
import pyarrow.dataset as ds
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


ENC_DICT = {
    "l": 1,
    "y": 2,
    "@": 3,
    "3": 4,
    "H": 5,
    "S": 6,
    "F": 7,
    "C": 8,
    "r": 9,
    "s": 10,
    "/": 11,
    "c": 12,
    "o": 13,
    "+": 14,
    "I": 15,
    "5": 16,
    "(": 17,
    "2": 18,
    ")": 19,
    "9": 20,
    "i": 21,
    "#": 22,
    "6": 23,
    "8": 24,
    "4": 25,
    "=": 26,
    "1": 27,
    "O": 28,
    "[": 29,
    "D": 30,
    "B": 31,
    "]": 32,
    "N": 33,
    "7": 34,
    "n": 35,
    "-": 36,
}
MAX_LENGTH = 142


class IterfeaturesDataset(IterableDataset):
    def __init__(self, encoded_path, batch_size=1500):
        self.encoded_path = encoded_path
        self.batch_size = batch_size
        self.total_rows = self._get_num_rows()

    def _get_num_rows(self):
        """Retrieve the number of rows from Parquet metadata"""
        dataset = ds.dataset(self.encoded_path, format="parquet")
        return dataset.count_rows()

    def __len__(self):
        """Return the total number of rows"""
        return self.total_rows

    def __iter__(self):
        """Stream Parquet data batch by batch"""
        dataset = ds.dataset(self.encoded_path, format="parquet")
        scanner = dataset.scanner(batch_size=self.batch_size)

        for batch in scanner.to_batches():
            df = batch.to_pandas()
            labels = df["label"].values
            encoded_smiles = df.drop(columns=["label"]).values

            for enc, lbl in zip(encoded_smiles, labels):
                assert lbl in [0, 1], f"Invalid label detected: {lbl}"
                yield torch.tensor(enc, dtype=torch.float32), torch.tensor(
                    lbl, dtype=torch.long
                )


class SmilesDataset(Dataset):
    """Custom Dataset for SMILES data"""

    def __init__(self, smiles, labels=None):
        """
        Args:
            smiles (pd.Series or list): List of SMILES strings
            labels (pd.Series or np.ndarray or None): Labels for the SMILES strings
        """
        self.smiles = smiles.tolist()
        if labels is not None:
            self.labels = (
                labels.values.astype(np.int64)
                if hasattr(labels, "values")
                else np.array(labels, dtype=np.int64)
            )
        else:
            self.labels = np.zeros(len(self.smiles), dtype=np.int64)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]


class SmilesEncDataset(Dataset):
    """for encoding SMILES"""

    def __init__(self, smiles, labels=None):
        """
        Args:
            smiles (pd.Series or list): List of SMILES strings
            labels (pd.Series or np.ndarray or None): Labels for the SMILES strings
        """
        self.smiles = smiles.tolist()
        if labels is not None:
            self.labels = (
                labels.values.astype(np.int64)
                if hasattr(labels, "values")
                else np.array(labels, dtype=np.int64)
            )
        else:
            self.labels = np.zeros(len(self.smiles), dtype=np.int64)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        encoded = [ENC_DICT.get(char, 0) for char in smile]
        padded = encoded + [0] * (MAX_LENGTH - len(encoded))
        return np.array(padded, dtype=np.int64), self.labels[idx]


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
        self.labels = (
            np.array(labels, dtype=np.int64)
            if labels is not None
            else np.zeros(len(self.smiles), dtype=np.int64)
        )
        self.fp_size = fp_size
        self.fpg = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=fp_size, includeChirality=True
        )
        self.bit_info = bit_info

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        label = self.labels[idx]

        # Convert SMILES to Mol object
        mol = Chem.MolFromSmiles(smiles.replace("[Dy]", "[H]"))
        if mol is None:
            fingerprint = np.zeros(self.fp_size, dtype=np.int8)
            bit_info = None
        else:
            if self.bit_info:
                ao = rdFingerprintGenerator.AdditionalOutput()
                ao.AllocateBitInfoMap()
                fp = self.fpg.GetCountFingerprint(mol, additionalOutput=ao)
                bit_info = ao.GetBitInfoMap()
            else:
                fp = self.fpg.GetFingerprint(mol)
                bit_info = None

            fingerprint = np.zeros(self.fp_size, dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, fingerprint)

        return (
            torch.tensor(fingerprint, dtype=torch.float32),
            bit_info,
            torch.tensor(label, dtype=torch.float32),
        )


def custom_collate_fn(batch):
    # Separate fingerprints, bit_info, and labels
    fingerprints = [item[0] for item in batch]
    bit_info = [item[1] for item in batch] if batch[0][1] is not None else None
    labels = [item[2] for item in batch]

    fingerprints = torch.stack(fingerprints)
    labels = torch.stack(labels)

    if bit_info is not None:
        return fingerprints, bit_info, labels
    else:
        return fingerprints, labels
