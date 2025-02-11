import pandas as pd 
import numpy as np
import pickle
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, MolFromSmiles, DataStructs
from rdkit.Chem import rdMolDescriptors
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBClassifier as xgb
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import streamlit as st
from rdkit.Chem import rdFingerprintGenerator


def load_model():
    with open("./lightgbm_cls.pkl", "rb") as f:
        model = pickle.load(f)
    return model

lightgbm_cls = load_model()





class Prediction:

    def __init__(self, data: pd.DataFrame):

        "Class initialization"

        self.data = data.copy()
        self.fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=700, useBondTypes=True, includeChirality=True, includeRingMembership=True)

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


    def compute_fn(self, smiles):
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(700, dtype=np.int8)
        
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        
        
        fp = self.fpg.GetCountFingerprint(mol, additionalOutput=ao)

        bit_info = ao.GetBitInfoMap()
        arr = np.zeros(700, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
    
        return arr, bit_info
    
    def feature_engineering(self, smiles_columns = ['molecule_smiles'], drop_columns=['id', 'molecule_smiles', 'buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'protein_name']):
        """
        Computes fingerprints for specified SMILES columns, concatenates them, and prepares feature matrix (X)

        Args:
            
            smiles_columns (list): List of column names containing SMILES strings
            drop_columns (list): List of columns to drop from the final feature matrix (default: ['id', 'molecule_smiles'])

        Returns:
            X (pd.DataFrame): Feature matrix containing concatenated fingerprints
        """

        results = self.data['molecule_smiles'].apply(self.compute_fn)
        self.data['fp'] = results.apply(lambda x: x[0])
        bit_infos = results.apply(lambda x: x[1]).tolist()
        self.data.drop(drop_columns, axis=1, inplace=True)
        fingerprint_df = pd.DataFrame(self.data["fp"].to_list(), index=self.data.index)
        self.data = pd.concat([self.data.drop(columns=["fp"]), fingerprint_df], axis=1)
        
        feature_cols = [col for col in self.data.columns if col not in drop_columns]
        X = self.data[feature_cols]

        return X, bit_infos

    
    def predict(self):


        self.data_validation()

        #Data transformation
        X, bit_infos = self.feature_engineering()

        y_pred = lightgbm_cls.predict(X)

        return y_pred, X, bit_infos


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
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            heavy_atoms = Descriptors.HeavyAtomCount(mol)

            return wt, log_p, tpsa, rotatable_bonds, heavy_atoms
        
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        
    def calculate_all_descriptors(self):

        """
        Calculate descriptors for each building block

        """

        # Calculate descriptors for each building block
        self.data['block_1_wt'], self.data['block_1_log_p'], self.data['block_1_tpsa'], \
        self.data['block_1_rotatable_bonds'], self.data['block_1_heavy_atoms'] = zip(*self.data['buildingblock1_smiles'].apply(self.calculate_descriptors))

        self.data['block_2_wt'], self.data['block_2_log_p'], self.data['block_2_tpsa'], \
        self.data['block_2_rotatable_bonds'], self.data['block_2_heavy_atoms'] = zip(*self.data['buildingblock2_smiles'].apply(self.calculate_descriptors))

        self.data['block_3_wt'], self.data['block_3_log_p'], self.data['block_3_tpsa'], \
        self.data['block_3_rotatable_bonds'], self.data['block_3_heavy_atoms'] = zip(*self.data['buildingblock3_smiles'].apply(self.calculate_descriptors))
        self.data['prediction'] = self.y_pred

    def plot_descriptor_distributions(self, output = 'streamlit'):

        """

        Plot distributions of descriptors for the molecule and each building block

        """

        descriptors = ['wt', 'log_p', 'tpsa', 'rotatable_bonds', 'heavy_atoms']
        blocks = ['block_1', 'block_2', 'block_3']

        for descriptor in descriptors:

            if output == 'streamlit':
                st.write(f"### {descriptor.capitalize()} Distributions")

            df_list = []
            for block in blocks:
                df_list.append(pd.DataFrame({
                    'Block': block,
                    descriptor: self.data[f'{block}_{descriptor}']
                }))
            df = pd.concat(df_list)

            fig = px.histogram(df, x=descriptor, color='Block', marginal="rug", nbins=30,
                               title=f'{descriptor.capitalize()} Distributions',
                               labels={descriptor: descriptor.capitalize()},
                               hover_data=df.columns)

            if output == 'streamlit':
                st.plotly_chart(fig)
            elif output == 'console':
                fig.show()


    def threshold_analysis(self, output='streamlit'):

        """

        Perform threshold analysis for each descriptor based on predictions

        """
        descriptors = ['wt', 'log_p', 'tpsa', 'rotatable_bonds', 'heavy_atoms']
        blocks = ['block_1', 'block_2', 'block_3']

        for descriptor in descriptors:

            if output == 'streamlit':
                st.write(f"### {descriptor.capitalize()} vs Prediction")

            df_list = []
            for block in blocks:
                df_list.append(pd.DataFrame({
                    'Block': block,
                    'Prediction': self.data['prediction'],
                    descriptor: self.data[f'{block}_{descriptor}']
                }))
            df = pd.concat(df_list)

            fig = px.box(df, x='Prediction', y=descriptor, color='Block',
                         title=f'{descriptor.capitalize()} vs Prediction',
                         labels={descriptor: descriptor.capitalize()},
                         hover_data=df.columns)

            if output == 'streamlit':
                st.plotly_chart(fig)
            elif output == 'console':
                fig.show()

    def plot_scatter_matrix(self, output='streamlit'):

        """

        Plot scatter plot matrix for descriptors

        """
        descriptors = ['wt', 'log_p', 'tpsa', 'rotatable_bonds', 'heavy_atoms']
        blocks = ['block_1', 'block_2', 'block_3']

        for block in blocks:
            if output == 'streamlit':
                st.write(f"### Scatter Plot Matrix for {block.capitalize()}")
            df = self.data[[f'{block}_{desc}' for desc in descriptors] + ['prediction']]
            df.columns = descriptors + ['prediction']
            fig = px.scatter_matrix(df, dimensions=descriptors, color='prediction', title=f'Scatter Plot Matrix for {block.capitalize()}')
            if output == 'streamlit':
                st.plotly_chart(fig)
            elif output == 'console':
                fig.show()

    def plot_3d_pca(self, output='streamlit'):
        """

        Plot 3D PCA for descriptors

        """

        descriptors = ['wt', 'log_p', 'tpsa', 'rotatable_bonds', 'heavy_atoms']
        blocks = ['block_1', 'block_2', 'block_3']

        for block in blocks:
            if output == 'streamlit':
                st.write(f"### 3D PCA for {block.capitalize()}")
            df = self.data[[f'{block}_{desc}' for desc in descriptors]]
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(df)
            df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
            df_pca['prediction'] = self.data['prediction']
            fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='prediction', title=f'3D PCA for {block.capitalize()}')
            if output == 'streamlit':
                st.plotly_chart(fig)
            elif output == 'console':
                fig.show()





    def run_analysis(self, output='streamlit'):
        """

        Run the full analysis pipeline

        """
        self.calculate_all_descriptors()
        self.plot_descriptor_distributions(output=output)
        self.threshold_analysis(output=output)
        self.plot_scatter_matrix(output=output)
        self.plot_3d_pca(output=output)









            
        