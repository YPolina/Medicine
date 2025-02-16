from rdkit.Chem import Descriptors, MolFromSmiles, rdMolDescriptors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import pandas as pd 
import numpy as np 

from scipy.stats import ttest_1samp, norm, ks_2samp
from statsmodels.stats.proportion import proportions_ztest

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


    def __init__(self, data, bits_info, model):
        """

        Args:

            data: pd.DataFrame - data containing SMILES of molecules, features(fingerprints), predicted_label and predicted probability
            bit_infos - information from fingerprints features (bonds, atoms)


        """

        self.data = data.copy()
        self.bits_info = bits_info
        self.model = model
        self.threshold = 0.5
        self.y_pred = self.data['Predicted Class']
        self.y_true = self.data['binds']
        self.y_pred_prob = self.data['Predicted Probability']
        
    def plot_probability_distribution(self):
        """
        Plot probability distribution for class 0 and class 1
        """
        fig = px.histogram(self.data, x="Predicted Probability", color="Predicted Class", nbins=50,
                           title=f"Probability Distribution for Class 0 and Class 1 (Threshold = {self.threshold})",
                           labels={"Predicted Probability": "Predicted Probability", "count": "Count"})
        fig.add_vline(x=self.threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold = {self.threshold}")
        st.plotly_chart(fig)

    def plot_class_counts(self):
        """
        Plot counts of predicted classes with a given threshold
        """
        class_counts = pd.Series(self.y_pred).value_counts().reset_index()
        class_counts.columns = ['Predicted Class', 'Count']

        fig = px.bar(class_counts, x='Predicted Class', y='Count',
                     title=f"Counts of Predicted Classes (Threshold = {self.threshold})",
                     labels={"Predicted Class": "Predicted Class", "Count": "Count"})
        st.plotly_chart(fig)

    def calculate_metrics(self):
        """
        Metric Calculations
        """
        accuracy = np.round(accuracy_score(self.y_true, self.y_pred), 2)
        precision = np.round(precision_score(self.y_true, self.y_pred), 2)
        recall = np.round(recall_score(self.y_true, self.y_pred), 2)
        f1 = np.round(f1_score(self.y_true, self.y_pred), 2)
        roc_auc = np.round(roc_auc_score(self.y_true, self.y_pred), 2)
        return accuracy, precision, recall, f1, roc_auc

    def bootstrap_metric(self, metric_func, n_bootstrap=100):
        """
        Generation of metric destribution using Bootstrap
        """
        metrics = []
        for _ in range(n_bootstrap):
            sample = self.data.sample(frac=1, replace=True)
            y_true_sample = sample['binds']
            y_pred_sample = sample['Predicted Class']
            metrics.append(metric_func(y_true_sample, y_pred_sample))
        return np.array(metrics)

    def perform_z_test(self, observed_value, expected_value, n):
        """
        Z-test
        """
        se = np.sqrt((expected_value * (1 - expected_value)) / n)
        z_score = (observed_value - expected_value) / se
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        return z_score, p_value

    def perform_t_test(self, observed_value, expected_value, bootstrap_distribution):

        """
        T-test
        """
        t_stat, p_value = ttest_1samp(bootstrap_distribution, expected_value)
        return t_stat, p_value

    def compare_metrics(self, expected_metrics, actual_metrics):
        """
        Metrics comparisom with stats tests
        
        Args:
            expected_metrics (list): (accuracy, precision, recall, f1)
            actual_metrics (list): (accuracy, precision, recall, f1)
        """
        z_score, p_value_z = self.perform_z_test(actual_metrics[0], expected_metrics[0], len(self.data))

        results = []
        for metric_name, observed_value, expected_value in zip(
            ["Accuracy", "Precision", "Recall", "F1"],
            actual_metrics,
            expected_metrics
        ):
            if metric_name == "Accuracy":
                test_statistic = z_score
                p_value = p_value_z
                test_type = "Z-test"
            else:
                bootstrap_distribution = self.bootstrap_metric(
                    precision_score if metric_name == "Precision" else
                    recall_score if metric_name == "Recall" else f1_score
                )
                test_statistic, p_value = self.perform_t_test(observed_value, expected_value, bootstrap_distribution)
                test_type = "T-test"

            results.append({
                "Metric": metric_name,
                "Expected Value": expected_value,
                "Observed Value": observed_value,
                "Test Statistic": test_statistic,
                "p-value": p_value,
                "Test Type": test_type,
                "Conclusion": "Reject H0" if p_value < 0.05 else "Fail to reject H0"
            })

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)
        return results_df

    def plot_model_statistics(self):
        
        #From validation set
        expected_metrics = [0.96, 0.74, 0.77, 0.75]

        st.header("Model Statistics ")
        st.subheader("Expected Model Metrics")
        st.write(f"""
        - **Accuracy**: {expected_metrics[0]}
        - **Precision**: {expected_metrics[1]}
        - **Recall**: {expected_metrics[2]}
        - **F1-Score**: {expected_metrics[3]}
        """)
        
        accuracy, precision, recall, f1, roc_auc = self.calculate_metrics()

        st.subheader("Metrics on Inference")
        st.write(f"""
        - **Accuracy**: {accuracy}
        - **Precision**: {precision}
        - **Recall**: {recall}
        - **F1-Score**: {f1}
        """)
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)
        fig = px.imshow(conf_matrix, labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Negative', 'Positive'], y=['Negative', 'Positive'])

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                fig.add_annotation(x=j, y=i, text=str(conf_matrix[i, j]), 
                                showarrow=False, font=dict(color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black'))

        st.plotly_chart(fig)
    
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_prob)
        fig = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                      title=f'ROC Curve (AUC = {roc_auc:.2f})')

        fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color='red', dash='dash'))
        fig.add_annotation(x=0.6, y=0.4, text=f'AUC = {roc_auc:.2f}', showarrow=False, font=dict(size=15))

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        st.plotly_chart(fig)

        st.subheader("Statistical Comparison of Metrics")
        results_df = self.compare_metrics(expected_metrics, [accuracy, precision, recall, f1])
        st.table(results_df)


    def run_ab_test(self):
        
        st.header("A/B Testing for Model Stability")

        test_size = st.slider("Select Test Size", min_value=100, max_value=len(self.data), value=500)
        
        train_data = self.data.iloc[:-test_size]
        test_data = self.data.iloc[-test_size:]

        train_metrics = self.calculate_metrics(train_data['binds'], train_data['Predicted Class'])
        test_metrics = self.calculate_metrics(test_data['binds'], test_data['Predicted Class'])

        st.subheader("Train Metrics")
        st.write(f"""
            - **Accuracy**: {train_metrics[0]}
            - **Precision**: {train_metrics[1]}
            - **Recall**: {train_metrics[2]}
            - **F1-Score**: {train_metrics[3]}
            """)

        st.subheader("Test Metrics")
        st.write(f"""
            - **Accuracy**: {test_metrics[0]}
            - **Precision**: {test_metrics[1]}
            - **Recall**: {test_metrics[2]}
            - **F1-Score**: {test_metrics[3]}
            """)
       
        st.subheader("Statistical Comparison of Metrics")
        self.compare_metrics(train_metrics, test_metrics)

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

    def visualize_distributions(self):

        self.calculate_all_descriptors()
        
        st.title("Molecular Descriptors Distribution Analysis")

        descriptors = ['wt', 'log_p', 'tpsa', 'rotatable_bonds', 'heavy_atoms']
        blocks = ['block_1', 'block_2', 'block_3']

        selected_desc = st.selectbox("Select Descriptor", descriptors)
        if selected_desc:
            st.write(f"### {selected_desc.capitalize()} Distribution")
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"{selected_desc.capitalize()} Distribution by Blocks", fontsize=16)

            for i, block in enumerate(blocks):
                col_name = f'{block}_{selected_desc}'
                sns.histplot(data=self.data, x=col_name, hue='binds', kde=True, ax=axes[i], palette='viridis')
                axes[i].set_title(f"{block} {selected_desc.capitalize()}")
                axes[i].set_xlabel(selected_desc.capitalize())
                axes[i].set_ylabel("Frequency")

            st.pyplot(fig)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"{selected_desc.capitalize()} Violin Plot by Blocks", fontsize=16)

            for i, block in enumerate(blocks):
                col_name = f'{block}_{selected_desc}'
                sns.violinplot(data=self.data, x='binds', y=col_name, ax=axes[i], hue='binds', palette='viridis', legend=False)
                axes[i].set_title(f"{block} {selected_desc.capitalize()}")
                axes[i].set_xlabel("Binds")
                axes[i].set_ylabel(selected_desc.capitalize())

            st.pyplot(fig)

    def perform_statistical_tests(self):
        st.title("Statistical Tests")

        descriptors = ['wt', 'log_p', 'tpsa', 'rotatable_bonds', 'heavy_atoms']
        blocks = ['block_1', 'block_2', 'block_3']

        results = []

       
        for block in blocks:
            for desc in descriptors:
                col_name = f'{block}_{desc}'

          
                bind_data = self.data[self.data['binds'] == 1][col_name].dropna()
                pred_1_data = self.data[self.data['prediction'] == 1][col_name].dropna()
                ks_stat_binds, p_value_binds = ks_2samp(bind_data, pred_1_data)

            
                pred_0_data = self.data[self.data['prediction'] == 0][col_name].dropna()
                no_bind_data = self.data[self.data['binds'] == 0][col_name].dropna()
                ks_stat_no_binds, p_value_no_binds = ks_2samp(pred_0_data, no_bind_data)

             
                def interpret_p_value(p_value):
                    if p_value < 0.05:
                        return "Reject H0: Distributions are different"
                    else:
                        return "Fail to reject H0: Distributions are similar"

            
                results.append({
                    "Block": block,
                    "Descriptor": desc,
                    "KS Statistic (Binds)": np.round(ks_stat_binds, 2),
                    "P-Value (Binds)": np.round(p_value_binds, 2),
                    "Conclusion (Binds)": interpret_p_value(p_value_binds),
                    "KS Statistic (No Binds)": np.round(ks_stat_no_binds, 2),
                    "P-Value (No Binds)": np.round(p_value_no_binds, 2),
                    "Conclusion (No Binds)": interpret_p_value(p_value_no_binds)
                })

    
        results_df = pd.DataFrame(results)

        with st.expander("### Show/Hide Full Results of Statistical Tests", expanded=False):
    
            st.write("### Filters")
            selected_blocks = st.multiselect(
                "Select Blocks to Display", 
                blocks, 
                default=blocks, 
                key="block_filter"
            )
            selected_descriptors = st.multiselect(
                "Select Descriptors to Display", 
                descriptors, 
                default=descriptors, 
                key="desc_filter"
            )

    
            filtered_results = results_df[
                (results_df['Block'].isin(selected_blocks)) & 
                (results_df['Descriptor'].isin(selected_descriptors))
            ]

        
            st.table(filtered_results)


        st.write("""
        **Interpretation:**
        - **H0 (Null Hypothesis):** The distributions are the same.
        - **P-Value < 0.05:** Reject H0, indicating that the distributions are significantly different.
        - **P-Value >= 0.05:** Fail to reject H0, indicating that the distributions are similar.
        """)


    def run_visualizations(self):
        """
        Run all visualizations in Streamlit
        """
        st.title("Classification Model Visualization")

        self.threshold = st.sidebar.slider("Select Probability Threshold", 0.0, 1.0, 0.5, 0.1)
        self.y_pred = (self.y_pred_prob >= self.threshold).astype(int)

        self.plot_probability_distribution()
        self.plot_class_counts()
        self.plot_model_statistics()
        self.visualize_distributions()
        self.perform_statistical_tests()
        #self.run_ab_test()
'''    

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
