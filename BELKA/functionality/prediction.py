import joblib
import json

import pandas as pd
import numpy as np
import torch
import gc
import os
from tqdm import tqdm
import functionality
from functionality.data_preparation import SmilesDataset, SmilesEncDataset, FingerprintDataset
from functionality.models import (
    ChemBertaBinaryClassifierLightning,
    CNNBinaryClassifierLightning,
    MolFormerXLBinaryClassifierLightning,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

import pyarrow.parquet as pq
import pyarrow as pa


class Predictor:

    def __init__(
        self,
        data: pd.DataFrame,
        model_dir="checkpoints/models/",
        feature_dir="/inference/embeddings/",
        save_dir="./inference/",
    ):
        """
        Initialize the model loader with a given directory for models.
        """
        self.data = data.copy()
        self.proteins = data.protein_name.unique()
        self.model_dir = model_dir
        self.feature_dir = feature_dir
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def compute_and_save_fingerprints(
        self, smiles, fp_path, bit_info_path, batch_size=1000
    ):
        """
        Compute fingerprints and bit information in batches and save them dynamically using Parquet

        Args:
            smiles (pd.Series): Data containing SMILES strings
            fp_path (str): Path to save computed fingerprints in Parquet format
            bit_info_path (str): Path to save bit information in JSON format
            batch_size (int): Number of SMILES strings to process per batch
        """
        dataset = FingerprintDataset(smiles.tolist(), bit_info=True)

        def custom_collate_fn(batch):
            fingerprints = [item[0] for item in batch]
            bit_info = [item[1] for item in batch] if batch[0][1] is not None else None
            labels = [item[2] for item in batch]

            fingerprints = torch.stack(fingerprints)
            labels = torch.stack(labels)

            if bit_info is not None:
                return fingerprints, bit_info, labels
            else:
                return fingerprints, labels

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=30,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        # Define Parquet schema (fingerprint features as int8)
        PARQUET_SCHEMA = pa.schema([pa.field(f"fp_{i}", pa.int8()) for i in range(700)])

        all_bit_info = []
        with self.s3.open(fp_path, "wb") as s3_file:
            with pq.ParquetWriter(
                s3_file, PARQUET_SCHEMA, compression="SNAPPY"
            ) as writer:
                for batch in tqdm(dataloader, desc="Computing Fingerprints"):

                    batch_fingerprints, batch_bit_info, _ = batch

                    batch_bit_info_json = [json.dumps(info) for info in batch_bit_info]
                    all_bit_info.extend(batch_bit_info_json)

                    df_batch = pd.DataFrame(
                        batch_fingerprints.numpy(),
                        columns=[f"fp_{i}" for i in range(700)],
                    )
                    table = pa.Table.from_pandas(df_batch, schema=PARQUET_SCHEMA)
                    writer.write_table(table)

                    del batch, batch_fingerprints, batch_bit_info, df_batch, table
                    gc.collect()

        # Save all bit_info to a JSON file
        with self.s3.open(bit_info_path, "w") as json_file:
            json.dump(all_bit_info, json_file)
        return f"Fingerprints saved to {fp_path}\nBit info saved to {bit_info_path}"

    def compute_embeddings(self, model_name, smiles, save_path, batch_size=1000):
        """
        Compute embeddings for a list of SMILES strings in batches and save them in parquet format

        Args:
            model_name (str): Pretrained model name
            smiles (pd.Series): Data containing SMILES strings
            batch_size (int): Number of SMILES strings to process per batch
        """
        if model_name == "ChemBert":
            model_path = "seyonec/PubChem10M_SMILES_BPE_450k"
        elif model_name == "MolFormer":
            model_path = "ibm/MoLFormer-XL-both-10pct"

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model = torch.compile(model)
        model.eval()

        dataset = SmilesDataset(smiles, None)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=30, shuffle=False
        )

        PARQUET_SCHEMA = pa.schema(
            [pa.field(str(i), pa.float32()) for i in range(model.config.hidden_size)]
        )

        with self.s3.open(save_path, "wb") as s3_file:
            with pq.ParquetWriter(
                s3_file, PARQUET_SCHEMA, compression="SNAPPY"
            ) as writer:
                for batch_smiles, _ in tqdm(dataloader, desc="Computing Embeddings"):
                    if model_name == "MolFormer":
                        tokens = tokenizer(
                            batch_smiles, padding=True, return_tensors="pt"
                        )
                    elif model_name == "ChemBert":
                        tokens = tokenizer(
                            batch_smiles,
                            padding=True,
                            truncation=True,
                            max_length=150,
                            return_tensors="pt",
                        )
                    tokens = {k: v.to(device) for k, v in tokens.items()}

                    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
                        outputs = model(**tokens)

                    if model_name == "ChemBert":
                        batch_embeddings = (
                            outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        )
                    elif model_name == "MolFormer":
                        batch_embeddings = outputs.pooler_output.cpu().numpy()

                    df_batch = pd.DataFrame(
                        batch_embeddings,
                        columns=[str(i) for i in range(batch_embeddings.shape[1])],
                    )
                    table = pa.Table.from_pandas(df_batch)

                    # Write batch to Parquet file
                    writer.write_table(table)

                # Clean up
                del batch_smiles, tokens, outputs, batch_embeddings
                gc.collect()
                torch.cuda.empty_cache()

        return f"Embeddings saved to {save_path}"

    def compute_encoded_smiles(self, smiles, save_path, batch_size=1000):
        """
        Compute encoded SMILES representations in batches and return them as numpy arrays.

        Args:
            smiles (pd.Series): Data containing SMILES strings
            batch_size (int): Number of SMILES strings to process per batch

        """
        dataset = SmilesEncDataset(smiles, None)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=30, shuffle=False
        )

        # 142 - max_length of the encoded features
        PARQUET_SCHEMA = pa.schema(
            [pa.field(f"char_{i}", pa.int64()) for i in range(142)]
        )
        MAX_LENGTH = 142

        with self.s3.open(save_path, "wb") as s3_file:
            with pq.ParquetWriter(
                s3_file, PARQUET_SCHEMA, compression="SNAPPY"
            ) as writer:
                for batch_smiles, _ in tqdm(dataloader, desc="Encoding SMILES"):
                    batch_smiles = np.stack(batch_smiles)

                    df_batch = pd.DataFrame(
                        batch_smiles, columns=[f"char_{i}" for i in range(MAX_LENGTH)]
                    )

                    table = pa.Table.from_pandas(df_batch, schema=PARQUET_SCHEMA)
                    writer.write_table(table)

                    gc.collect()

        return f"Encoded SMILES saved to {save_path}"

    def feature_engineering(self, protein_data, protein_name):

        path = f"{self.path}/inference/embeddings"

        # Fingerprints generation
        fingerprint_path = os.path.join(path, f"{protein_name}_lightgbm_fp.parquet")
        bit_info_path = os.path.join(path, f"{protein_name}_bit_info.parquet")
        self.compute_and_save_fingerprints(
            protein_data["molecule_smiles"], fingerprint_path, bit_info_path
        )

        # ChemBert embeddings
        chembert_emb_path = os.path.join(path, f"{protein_name}_ChemBert_emb.parquet")
        self.compute_embeddings(
            "ChemBert", protein_data["molecule_smiles"], chembert_emb_path
        )

        # MolFormer embeddings
        molformer_emb_path = os.path.join(path, f"{protein_name}_MolFormer_emb.parquet")
        self.compute_embeddings(
            "MolFormer", protein_data["molecule_smiles"], molformer_emb_path
        )

        # encoded smiles
        enc_path = os.path.join(path, f"{protein_name}_CNN_enc.parquet")
        self.compute_encoded_smiles(protein_data["molecule_smiles"], enc_path)

    def load_torch_model(self, model_class, model_filename, device="cuda"):
        """
        Load a PyTorch model from a local file

        Args:
            model_class: The class of the model to be instantiated
            model_filename: The name of the model checkpoint file
            device: The device to load the model on ('cuda' or 'cpu')

        Returns:
            A loaded PyTorch model
        """
        model_path = os.path.join(self.model_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        print(f"Model {model_filename} loaded successfully on {device}!")
        return model

    def load_lightgbm_model(self, model_filename):
        """
        Load a LightGBM model from a local file

        Args:
            model_filename: The name of the LightGBM model file

        Returns:
            A loaded LightGBM model
        """
        model_path = os.path.join(self.model_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LightGBM model file not found: {model_path}")

        model = joblib.load(model_path)
        print(f"LightGBM model {model_filename} loaded successfully!")
        return model

    def load_all_models(self, protein_name, device="cuda"):
        """
        Load all models (CNN, ChemBERTa, MolFormer, and LightGBM) for a given protein name

        Args:
            protein_name: The name of the protein to load models for
            device: The device to load PyTorch models on ('cuda' or 'cpu')

        Returns:
            Tuple containing the loaded models (CNN, ChemBERTa, MolFormer, LightGBM)
        """
        cnn_model = self.load_torch_model(
            CNNBinaryClassifierLightning, f"{protein_name}_CNN.ckpt", device
        )
        chembert = self.load_torch_model(
            ChemBertaBinaryClassifierLightning, f"{protein_name}_ChemBert.ckpt", device
        )
        molformer = self.load_torch_model(
            MolFormerXLBinaryClassifierLightning,
            f"{protein_name}_MolFormer.ckpt",
            device,
        )
        lightgbm_model = self.load_lightgbm_model(f"{protein_name}_lightgbm.pkl")

        return cnn_model, chembert, molformer, lightgbm_model

    def load_features(self, model_name, protein_name):
        """
        Load features from a parquet file

        Args:
            model_name (str): Model type
            protein_name (str): Protein name for feature file

        Returns:
            np.ndarray: Feature matrix
        """
        file_path = os.path.join(
            self.feature_dir, f"{protein_name}_{model_name}.parquet"
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Feature file not found: {file_path}")

        table = pq.read_table(file_path)
        return table.to_pandas().values

    def predict(
        self, data, proteins, save_filename="predictions.parquet", device="cuda"
    ):
        """
        Perform prediction using pre-trained models and save results

        Args:
            data (pd.DataFrame): Input data containing 'protein_name' column
            proteins (list): List of proteins to process
            save_filename (str): Filename to save predictions
            device (str): Device for PyTorch models

        Returns:
            pd.DataFrame: Dataframe with predictions
        """
        results = data.copy()
        results["predict_probability"] = np.nan

        for protein in proteins:
            print(f"Loading pre-trained models for {protein}")
            cnn_model, chembert_model, molformer_model, lightgbm_model = (
                self.load_all_models(protein, device)
            )
            print(f"Models loaded successfully for protein {protein}")

            print(f"Feature engineering for protein {protein}")
            protein_data = results[results["protein_name"] == protein]
            self.feature_engineering(protein_data, protein)
            print(f"Features generated for protein {protein} successfully")

            cnn_features = self.load_features("CNN_enc", protein)
            chembert_features = self.load_features("ChemBert_emb", protein)
            molformer_features = self.load_features("MolFormer_emb", protein)
            lightgbm_features = self.load_features("lightgbm_fp", protein)

            meta_model_path = os.path.join(
                self.model_dir, f"{protein}_random_forest.pkl"
            )
            if not os.path.exists(meta_model_path):
                raise FileNotFoundError(f"Meta model file not found: {meta_model_path}")

            meta_model = joblib.load(meta_model_path)

            new_meta_features = np.column_stack(
                (
                    cnn_model.predict_proba(cnn_features),
                    chembert_model.predict_proba(chembert_features),
                    molformer_model.predict_proba(molformer_features),
                    lightgbm_model.predict_proba(lightgbm_features)[
                        :, 1
                    ],  
                )
            )
            

            # Use the meta-model to make final predictions
            final_probabilities = meta_model.predict_proba(new_meta_features)[:, 1]
            final_labels = meta_model.predict(new_meta_features)[:, 1]

            results.loc[protein_data.index, "Predicted Probability"] = final_probabilities
            results.loc[protein_data.index, "Predicted Class"] = final_labels

            # Save predictions
            save_path = os.path.join(self.save_dir, save_filename)
            results[["id", "predict_probability"]].to_parquet(save_path)
            print(f"Predictions saved to {save_path}")

            return results
