import joblib
import json
import os
os.environ["TRUST_REMOTE_CODE"] = "True"
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from datasets import Dataset as HFDataset
from functionality.data_preparation import CustomDataset, FingerprintDataset
from functionality.models import (
    ChemBertBinaryClassifier,
    MolFormerClassifier,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding



class Predictor:

    def __init__(
        self,
        model_dir="./trained_models/",
    ):
        """
        Initialize the model loader with a given directory for models.
        """
        self.model_dir = model_dir
        self.num_workers = torch.cuda.device_count() * 8 if torch.cuda.is_available() else 8

    def fp_prediction(self, smiles, ids, model, batch_size=300):
        """
        Compute fingerprints, predict probabilities, and save bit_info in JSON format.

        Args:
            smiles (pd.Series): SMILES strings
            ids (pd.Series): Unique IDs corresponding to each SMILES string
            batch_size (int): Batch size for DataLoader

        Returns:
            np.ndarray: Predicted probabilities for the positive class
        """

        # Create the dataset
        dataset = FingerprintDataset(smiles.tolist(), bit_info=True)

        def custom_collate_fn(batch):
            fingerprints = [item[0] for item in batch]
            bit_info = [item[1] for item in batch]
            labels = [item[2] for item in batch]
            fingerprints = torch.stack(fingerprints)
            labels = torch.stack(labels)
            
            return fingerprints, bit_info, labels

        # Create the DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=custom_collate_fn
        )

        # Initialize lists to store results
        all_probs = []
        combined_bit_info = {}

        # Predict in batches
        for i, (X_batch, bit_info_batch, _) in enumerate(tqdm(dataloader, desc="Prediction with lighgbm")):
            X_batch = X_batch.numpy()

            num_features = X_batch.shape[1]
            df_X_batch = pd.DataFrame(X_batch, columns=[f"feature_{j}" for j in range(num_features)])

            batch_probs = model.predict_proba(df_X_batch)[:, 1] 
            all_probs.append(batch_probs)

            # Save bit_info with corresponding IDs
            batch_ids = ids.iloc[i * batch_size:(i + 1) * batch_size].tolist()
            for j, info in enumerate(bit_info_batch):
                combined_bit_info[batch_ids[j]] = info

            del df_X_batch, batch_probs, batch_ids, bit_info_batch

        # Use the only available batch instead of concatenating
        if len(all_probs) == 1:
            all_probs = all_probs[0]
        # Normal case for multiple batches
        elif len(all_probs) > 1:
            all_probs = np.concatenate(all_probs) 
        else:
            all_probs = np.array([]) 

        # Save bit_info to a JSON file
        with open("bit_info.json", 'w') as json_file:
            json.dump(combined_bit_info, json_file)

        print(f"Bit info saved to bit_info.json")
        return all_probs

    def emb_prediction(self, model_name, data, model, batch_size=1000):
        """
        Generates molecular embeddings using a specified deep learning model

        Parameters:
        - model_name (str): The name of the model to be used ('ChemBert' or 'MolFormer')
        - data (pandas.DataFrame): The input dataset containing molecular SMILES strings
        - model (torch.nn.Module): The preloaded deep learning model for generating embeddings
        - batch_size (int, optional): The batch size for processing the data (default: 1000)

        Returns:
        - y_proba (list): A list of predicted probabilities or embeddings for the input molecules
        """
        
    
        # Load the tokenizer
        if model_name == 'ChemBert':
            model_path = "seyonec/PubChem10M_SMILES_BPE_450k"
        elif model_name == 'MolFormer':
            model_path = "ibm/MoLFormer-XL-both-10pct"
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        MAX_LENGTH = 150

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Tokenization function
        def tokenize_function(samples, tokenizer):
            tokenized = tokenizer(
                samples['molecule_smiles'],
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH
            )
            return tokenized
        dataset = HFDataset.from_pandas(data)


        dataset_tokenized = dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
        dataset_custom = CustomDataset(dataset_tokenized, include_labels=False)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        dataloader = DataLoader(
            dataset_custom,
            batch_size=batch_size,
            num_workers=self.num_workers,
            collate_fn=data_collator
        )

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        y_proba = []

        with torch.no_grad(): 
            for batch in tqdm(dataloader, desc=f'Prediction with {model_name}'):
                batch = {k: v.to(device) for k, v in batch.items()}
                probs = model.predict_step(batch, batch_idx=None)  
                y_proba.extend(probs.cpu().numpy())
        return y_proba


    def model_load(self, protein_name):
        """
        Loads pre-trained models for molecular prediction tasks

        Parameters:
        - protein_name (str): The name of the target protein for which models are being loaded

        Returns:
        - chembert_model (torch.nn.Module): The loaded ChemBert model
        - molformer_model (torch.nn.Module): The loaded MolFormer model
        - lightgbm_model (LightGBM model): The loaded LightGBM model
        - meta_model (scikit-learn model): The loaded meta-model
        """
        
    
        chembert_model_path = f'{self.model_dir}{protein_name}_ChemBert.ckpt'
        molformer_model_path = f'{self.model_dir}{protein_name}_MolFormer.ckpt'
    
        lightgbm_model_path = f'{self.model_dir}{protein_name}_lightgbm.pkl'
        meta_model_path = f'{self.model_dir}{protein_name}_meta_model.pkl'

        chembert_model = ChemBertBinaryClassifier.load_from_checkpoint(chembert_model_path)
        molformer_model = MolFormerClassifier.load_from_checkpoint(molformer_model_path, trust_remote_code=True)

        lightgbm_model = joblib.load(lightgbm_model_path)
        meta_model = joblib.load(meta_model_path)

        return chembert_model, molformer_model, lightgbm_model, meta_model


    def prediction_and_bit_info(self, model_input: pd.DataFrame):
        """
        Perform prediction using pre-trained models and save results

        Args:
            model_input (pd.DataFrame): Input data containing 'protein_name' column
           

        Returns:
            pd.DataFrame: Dataframe with predictions, bit_info - dictionary with information from fingerprins features 
        """
        results = model_input.copy()
        if 'id' not in results:
            results['id'] = results.get('id', results.index)
        results["Predicted Probability"] = np.nan

        for protein in results.protein_name.unique():

            protein_data = results[results['protein_name'] == protein]
            print(f"Loading pre-trained models for {protein}")
            chembert_model, molformer_model, lightgbm_model, meta_model = (
                self.model_load(protein)
            )
            print(f"Models loaded successfully for protein {protein}")

            chembert_prob = self.emb_prediction('ChemBert', protein_data, chembert_model)
            molformer_prob = self.emb_prediction('MolFormer', protein_data, molformer_model)
            lightgbm_prob = self.fp_prediction(protein_data['molecule_smiles'], protein_data['id'], lightgbm_model)

            new_meta_features = np.column_stack(
                (
                    chembert_prob,
                    molformer_prob,
                    lightgbm_prob,  
                )
            )

            np.set_printoptions(precision=5)

            final_probabilities = meta_model.predict_proba(new_meta_features)[:, 1]
            final_probabilities = np.round(final_probabilities, 5)
            final_labels = meta_model.predict(new_meta_features)

            results.loc[protein_data.index, "Predicted Probability"] = final_probabilities
            results.loc[protein_data.index, "Predicted Class"] = final_labels

        with open('bit_info.json') as f:
            bit_info = json.load(f)


        return results, bit_info 
    

        
