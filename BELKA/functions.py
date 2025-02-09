import pandas as pd 
import numpy as np
import pickle
import torch
from rdkit.Chem import Descriptors, MolFromSmiles
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBClassifier as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import streamlit as st

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

@st.cache_resource
def load_model():
    with open("./xgb_cls.pkl", "rb") as f:
        model = pickle.load(f)
    return model

xgb_cls = load_model()


class Prediction:

    def __init__(self, data: pd.DataFrame):

        "Class initialization"

        self.data = data.copy()

    def data_validation(self):
        """
        Input check for format corresponding

        """
        #Check for nessesary columns
        required_columns = ['id', 'buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles', 'protein_name']
        for col in self.data.columns:
            if col not in required_columns:
                raise ValueError(f'Column {col} not expected.')
        
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f'Column {col} is missing in input data')
            

        #Validate protein name
        if not self.data['protein_name'].eq('sEH').all():
            raise ValueError('Protein name must be sEH')
            
        #NaN values check
        if self.data.isna().any(axis=None):
            self.data = self.data.dropna()
            print('Input data contains NaN values, that have been dropped successfully.')
        
        #Validate id type
        if pd.api.types.is_integer_dtype(self.data['id']):
            try:
                self.data['id'] = self.data['id'].apply(int)
            except Exception as e:
                print(f'{e}. \n ID column must contain integer or converting to integer dtype.')


        #Validate Smiles columns    
        smiles_columns = ['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles']
        for col in smiles_columns:
            if not self.data[col].apply(lambda x: self._is_valid_smiles(x)).all():
                 raise ValueError(f'Column {col} contains invalid SMILES strings')
            

    def _is_valid_smiles(self, smiles):

        "Helper function to check whether the input smiles is valid"

        if pd.isna(smiles):
            return False
        mol = MolFromSmiles(smiles)

        return mol is not None
    
    def smiles_embeddings(self, smiles:str):

        """
        Smiles to BERT Embeddings
        
        """

        tokens = tokenizer(smiles, padding = True, truncation = True, max_length = 512, return_tensors = 'pt')
        
        with torch.no_grad():
            outputs = model(**tokens)
        
        return outputs.last_hidden_state.mean(dim = 1).squeeze().cpu().numpy().tolist()

    
    def predict(self):


        self.data_validation()


        #Bert Embeddings for the input
        self.data['buildingblock1_embedding'] = self.data['buildingblock1_smiles'].apply(self.smiles_embeddings)
        self.data['buildingblock2_embedding'] = self.data['buildingblock2_smiles'].apply(self.smiles_embeddings)
        self.data['buildingblock3_embedding'] = self.data['buildingblock3_smiles'].apply(self.smiles_embeddings)
        self.data['molecule_embedding'] = self.data['molecule_smiles'].apply(self.smiles_embeddings)

        #Data transformation
        X = self.data[['buildingblock1_embedding', 'buildingblock2_embedding', 'buildingblock3_embedding', 'molecule_embedding']]

        X_test = np.array(X.map(lambda x: np.array(x).tolist()).values.tolist())
        X_test = np.array([np.concatenate(row) for row in X_test])

        y_pred = xgb_cls.predict(X_test.tolist())
        y_pred_proba = xgb_cls.predict_proba(X_test.tolist())[:, 1]

        return y_pred, y_pred_proba, self.data


class Analysis:

    '''
    Descriptors analysis based on predictions
    '''


    def __init__(self, data, y_pred):
        """

        Args:

            data: pd.DataFrame - Data containing SMILES string and Embeddings
            y_pred: np.array - predicted labels from the model

        """

        self.data = data.copy()
        self.y_pred = y_pred

    def calculate_descriptors(self, smiles):

        """
        Calculate Molecular Descriptors for a given SMILES string

        Args:

            smiles:str - SMILES string

        Returns:

            tuple: Molecular weight, logP, and TPSA or (NaN, NaN, NaN) if SMILES is invalid.
        """

        mol = MolFromSmiles(smiles)

        if mol is not None:

            wt = Descriptors.MolWt(mol)
            log_p = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)

            return wt, log_p, tpsa
        
        else:
            return np.nan, np.nan, np.nan
        
    def calculate_all_descriptors(self):

        """
        Calculate descriptors for the molecule and each building block

        """

        # Calculate descriptors for the molecule
        self.data['mol_wt'], self.data['mol_log_p'], self.data['mol_tpsa'] = zip(*self.data['molecule_smiles'].apply(self.calculate_descriptors))

        # Calculate descriptors for each building block
        self.data['block_1_wt'], self.data['block_1_log_p'], self.data['block_1_tpsa'] = zip(*self.data['buildingblock1_smiles'].apply(self.calculate_descriptors))
        self.data['block_2_wt'], self.data['block_2_log_p'], self.data['block_2_tpsa'] = zip(*self.data['buildingblock2_smiles'].apply(self.calculate_descriptors))
        self.data['block_3_wt'], self.data['block_3_log_p'], self.data['block_3_tpsa'] = zip(*self.data['buildingblock3_smiles'].apply(self.calculate_descriptors))

        self.data['prediction'] = self.y_pred

    def plot_descriptor_distributions(self, output = 'streamlit'):

        """

        Plot distributions of descriptors for the molecule and each building block

        """

        descriptors = ['wt', 'log_p', 'tpsa']
        blocks = ['mol', 'block_1', 'block_2', 'block_3']

        for descriptor in descriptors:

            if output == 'streamlit':
                st.write(f"### {descriptor.capitalize()} Distributions")

            plt.figure(figsize = (15, 10))

            for i, block in enumerate(blocks):

                fig, ax = plt.subplots()
                sns.histplot(self.data[f'{block}_{descriptor}'], kde=True, bins=30, ax=ax)
                ax.set_title(f'{block.capitalize()} {descriptor.capitalize()} Distribution')
                ax.set_xlabel(f'{descriptor.capitalize()}')
                ax.set_ylabel('Frequency')

            if output == 'streamlit':
                st.pyplot(fig)
            elif output == 'console':
                plt.show()

            #plt.close(fig)


    def threshold_analysis(self, output='streamlit'):

        """

        Perform threshold analysis for each descriptor based on predictions

        """
        descriptors = ['wt', 'log_p', 'tpsa']
        blocks = ['mol', 'block_1', 'block_2', 'block_3']

        for descriptor in descriptors:

            if output == 'streamlit':
                st.write(f"### {descriptor.capitalize()} vs Prediction")

            for block in blocks:

                fig, ax = plt.subplots()
                sns.boxplot(x='prediction', y=f'{block}_{descriptor}', data=self.data, ax=ax)
                ax.set_title(f'{block.capitalize()} {descriptor.capitalize()} vs Prediction')
                ax.set_xlabel('Prediction (0 or 1)')
                ax.set_ylabel(f'{descriptor.capitalize()}')

        if output == 'streamlit':
            st.pyplot(fig)
        elif output == 'console':
            plt.show()
        #plt.close(fig)

    def run_analysis(self, output='streamlit'):
        """

        Run the full analysis pipeline

        """
        self.calculate_all_descriptors()
        self.plot_descriptor_distributions(output=output)
        self.threshold_analysis(output=output)









            
        