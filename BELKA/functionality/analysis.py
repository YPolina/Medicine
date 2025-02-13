from rdkit.Chem import Descriptors, MolFromSmiles, rdMolDescriptors
import pandas as pd 
import numpy as np 

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.decomposition import PCA
import streamlit as st

class Analyser:

    '''
    Descriptors analysis based on predictions
    '''


    def __init__(self, data, bits_info):
        """

        Args:

            data: pd.DataFrame - data containing SMILES of molecules, features(fingerprints), predicted_label and predicted probability
            bit_infos - information from fingerprints features (bonds, atoms)


        """

        self.data = data.copy()
        self.bits_info = bits_info
        
    def plot_probability_distribution(self):
        """Plot probability distribution for class 0 and class 1."""
        fig = px.histogram(self.data, x="Predicted Probability", color="Predicted Class", nbins=50,
                            title="Probability Distribution for Class 0 and Class 1",
                            labels={"Predicted Probability": "Predicted Probability", "count": "Count"})
        st.plotly_chart(fig)

    def plot_class_counts(self, threshold=0.5):
        """Plot counts of predicted classes with a given threshold."""
        self.data['Predicted Class with Threshold'] = (self.data['Predicted Probability'] >= threshold).astype(int)
        class_counts = self.data['Predicted Class with Threshold'].value_counts().reset_index()
        class_counts.columns = ['Predicted Class', 'Count']

        fig = px.bar(class_counts, x='Predicted Class', y='Count',
                     title=f"Counts of Predicted Classes (Threshold = {threshold})",
                     labels={"Predicted Class": "Predicted Class", "Count": "Count"})
        st.plotly_chart(fig)


    def plot_advanced_threshold_range(self, threshold_range=(0.3, 0.7)):
        """Plot counts of predicted classes within a threshold range."""
        self.data['Predicted Class with Range'] = ((self.data['Predicted Probability'] >= threshold_range[0]) & 
                                                   (self.data['Predicted Probability'] <= threshold_range[1])).astype(int)
        range_counts = self.data['Predicted Class with Range'].value_counts().reset_index()
        range_counts.columns = ['Predicted Class', 'Count']

        fig = px.bar(range_counts, x='Predicted Class', y='Count',
                       title=f"Counts of Predicted Classes (Threshold Range = {threshold_range})",
                       labels={"Predicted Class": "Predicted Class", "Count": "Count"})
        
        st.plotly_chart(fig)

    def display_model_statistics(self):
        """Display predefined model statistics."""
        st.header("Model Statistics and A/B Testing")
        st.subheader("Expected Model Metrics")
        st.write("""
        - **Accuracy**: 0.85
        - **Precision**: 0.83
        - **Recall**: 0.87
        - **F1-Score**: 0.85
        """)

        st.subheader("A/B Testing")
        st.write("""
        - **Trained on Dataset Size (X)**: 10,000 samples
        - **Tested on Dataset Size (Y)**: 2,000 samples
        - **Current Evaluation Size**: 1,000 samples
        - **Confidence Interval**: 95%
        - **Expected Metric Stability**: ±2%
        """)

    def run_visualizations(self):
        """Run all visualizations in Streamlit."""
        
        st.title("Classification Model Visualization")

        # Sidebar for threshold selection
        threshold = st.sidebar.slider("Select Probability Threshold", 0.0, 1.0, 0.5, 0.1)
        threshold_range = st.sidebar.slider("Select Probability Threshold Range", 0.0, 1.0, (0.3, 0.7), 0.1)

        # Visualizations
        self.plot_probability_distribution()
        self.plot_class_counts(threshold)
        self.plot_advanced_threshold_range(threshold_range)
        self.display_model_statistics()
'''    

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

    def plot_descriptor_distributions(self):

        """

        Plot distributions of descriptors for the molecule and each building block

        0"""

        descriptors = ['wt', 'log_p', 'tpsa', 'rotatable_bonds', 'heavy_atoms']
        blocks = ['block_1', 'block_2', 'block_3']

        for descriptor in descriptors:

            if self.output == 'streamlit':
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

            if self.output == 'streamlit':
                st.plotly_chart(fig)
            elif self.output == 'console':
                fig.show()


    def threshold_analysis(self):

        """

        Perform threshold analysis for each descriptor based on predictions

        """
        descriptors = ['wt', 'log_p', 'tpsa', 'rotatable_bonds', 'heavy_atoms']
        blocks = ['block_1', 'block_2', 'block_3']

        for descriptor in descriptors:

            if self.output == 'streamlit':
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

            if self.output == 'streamlit':
                st.plotly_chart(fig)
            elif self.output == 'console':
                fig.show()

    def plot_scatter_matrix(self):

        """

        Plot scatter plot matrix for descriptors

        """
        descriptors = ['wt', 'log_p', 'tpsa', 'rotatable_bonds', 'heavy_atoms']
        blocks = ['block_1', 'block_2', 'block_3']

        for block in blocks:
            if self.output == 'streamlit':
                st.write(f"### Scatter Plot Matrix for {block.capitalize()}")
            df = self.data[[f'{block}_{desc}' for desc in descriptors] + ['prediction']]
            df.columns = descriptors + ['prediction']
            fig = px.scatter_matrix(df, dimensions=descriptors, color='prediction', title=f'Scatter Plot Matrix for {block.capitalize()}')
            if self.output == 'streamlit':
                st.plotly_chart(fig)
            elif self.output == 'console':
                fig.show()

    def plot_3d_pca(self):
        """

        Plot 3D PCA for descriptors

        """

        descriptors = ['wt', 'log_p', 'tpsa', 'rotatable_bonds', 'heavy_atoms']
        blocks = ['block_1', 'block_2', 'block_3']

        for block in blocks:
            if self.output == 'streamlit':
                st.write(f"### 3D PCA for {block.capitalize()}")
            df = self.data[[f'{block}_{desc}' for desc in descriptors]]
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(df)
            df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
            df_pca['prediction'] = self.data['prediction']
            fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='prediction', title=f'3D PCA for {block.capitalize()}')
            if self.output == 'streamlit':
                st.plotly_chart(fig)
            elif self.output == 'console':
                fig.show()





    def run_analysis(self):
        """

        Run the full analysis pipeline

        """
        self.probability_distribution()
        #self.calculate_all_descriptors()
        #self.plot_descriptor_distributions()
        #self.threshold_analysis()
        #self.plot_scatter_matrix()
        #self.plot_3d_pca()
    '''
