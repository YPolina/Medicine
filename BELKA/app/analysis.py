from rdkit.Chem import Descriptors, MolFromSmiles, rdMolDescriptors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import pandas as pd 
import numpy as np 
from scipy.stats import ttest_1samp, norm, ks_2samp
from pathlib import Path
from utils import invoke_endpoint_decorator
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

class Analyser:
    def __init__(self, data, bit_info):
        """
        Initializes the Analyser class

        Args:
            data (pd.DataFrame): Data containing SMILES of molecules, features (fingerprints), 
                                 predicted label, and predicted probability
            bit_info (dict): Information about the bits in molecular fingerprints
        """

        self.data = data.copy()
        self.previous_test_size = None
        self.bits_info = bit_info
        self.threshold = 0.5
        self.y_pred = self.data['Predicted Class']
        self.y_true = self.data['binds'] if 'binds' in data else None
        self.y_pred_prob = self.data['Predicted Probability']
        self.descriptors = ['wt', 'log_p', 'tpsa', 'rotatable_bonds', 'heavy_atoms']
        self.palette = {'0': '#1f77b4', '1': '#ff7f0e'}
        self.data['binds_plot'] = self.data['binds'].astype(str)
        
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

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculates classification metrics

        Args:
            y_true (array-like): True class labels
            y_pred (array-like): Predicted class labels
        
        Returns:
            tuple: Accuracy, precision, recall, F1-score, and ROC AUC score
        """
        accuracy = np.round(accuracy_score(y_true, y_pred), 2)
        precision = np.round(precision_score(y_true, y_pred), 2)
        recall = np.round(recall_score(y_true, y_pred), 2)
        f1 = np.round(f1_score(y_true, y_pred), 2)
        roc_auc = np.round(roc_auc_score(y_true, y_pred), 2)
        return accuracy, precision, recall, f1, roc_auc

    def bootstrap_metric(self, metric_func, n_bootstrap=100):
        """
        Generates a metric distribution using bootstrapping

        Args:
            metric_func (function): Metric function to compute
            n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 100
        
        Returns:
            np.array: Array of computed metric values
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
        Performs a Z-test for comparing proportions

        Args:
            observed_value (float): Observed metric value
            expected_value (float): Expected metric value
            n (int): Sample size
        
        Returns:
            tuple: Z-score and p-value
        """
        se = np.sqrt((expected_value * (1 - expected_value)) / n)
        z_score = (observed_value - expected_value) / se
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        return z_score, p_value

    def perform_t_test(self, observed_value, expected_value, bootstrap_distribution):
        """
        Performs a T-test using a bootstrap distribution

        Args:
            observed_value (float): Observed metric value
            expected_value (float): Expected metric value
            bootstrap_distribution (array-like): Bootstrap samples
        
        Returns:
            tuple: T-statistic and p-value
        """
        t_stat, p_value = ttest_1samp(bootstrap_distribution, expected_value)
        return t_stat, p_value

    def compare_metrics(self, expected_metrics, actual_metrics):
        """
        Compares model performance metrics using statistical tests

        Args:
            expected_metrics (list): Expected accuracy, precision, recall, F1-score
            actual_metrics (list): Actual accuracy, precision, recall, F1-score
        
        Returns:
            pd.DataFrame: Results of statistical tests
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
        """
        Plots various model evaluation statistics, including confusion matrix and ROC curve
        """
        
        #From validation set
        expected_metrics = [0.96, 0.70, 0.875, 0.78]

        st.header("Model Statistics")
        st.subheader("Expected Model Metrics")
        st.write(f"""
        - **Accuracy**: {expected_metrics[0]}
        - **Precision**: {expected_metrics[1]}
        - **Recall**: {expected_metrics[2]}
        - **F1-Score**: {expected_metrics[3]}
        """)
        
        accuracy, precision, recall, f1, roc_auc = self.calculate_metrics(self.y_true, self.y_pred)

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

    @invoke_endpoint_decorator
    def get_ab_test_predictions(self, results_df, bit_info):
        return results_df, bit_info
    
    def run_ab_test(self):

        """
        Conducts an A/B test on model performance across different data subsets
        """
        
        st.header("A/B Testing for Model Stability")

        test_size = st.slider("Select Test Size", min_value=100, max_value=len(self.data), value=len(self.data))
        train_metrics = self.calculate_metrics(self.y_true, self.y_pred)
        st.subheader("Input data metrics")
        st.write(f"""
            - **Accuracy**: {train_metrics[0]}
            - **Precision**: {train_metrics[1]}
            - **Recall**: {train_metrics[2]}
            - **F1-Score**: {train_metrics[3]}
            """)
        if test_size != len(self.data):
            test_data = self.data.iloc[-test_size:]

            try:
                results_2, _ = self.get_ab_test_predictions(test_data)
            except Exception as e:
                st.error(f"Failed to invoke endpoint for A/B testing: {e}")
                st.stop()

            test_metrics = self.calculate_metrics(results_2['binds'], results_2['Predicted Class'])

            st.subheader("Metrics on Part of Test Set")
            st.write(f"""
                - **Accuracy**: {test_metrics[0]}
                - **Precision**: {test_metrics[1]}
                - **Recall**: {test_metrics[2]}
                - **F1-Score**: {test_metrics[3]}
            """)

            st.subheader("Statistical Comparison of Metrics")
            self.compare_metrics(train_metrics, test_metrics)
            
            self.previous_test_size = test_size

    def calculate_descriptors(self, smiles):
        """
        Calculates molecular descriptors from a given SMILES string

        Args:
            smiles (str): SMILES representation of a molecule
        
        Returns:
            tuple: Molecular weight, logP, TPSA, rotatable bonds, heavy atoms count
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
        Computes descriptors for all molecules in the dataset and updates the dataframe
        """
        ...
        #Calculate descriptoes for the whole molecule
        self.data['mol_wt'], self.data['mol_log_p'], self.data['mol_tpsa'], \
        self.data['mol_rotatable_bonds'], self.data['mol_heavy_atoms'] = zip(*self.data['molecule_smiles'].apply(self.calculate_descriptors))

        self.data['prediction'] = self.y_pred

    def visualize_distributions(self, protein_name):
        """
        Visualizes molecular descriptor distributions for a specific protein.

        Args:
            protein_name (str): Name of the protein
        """
        self.calculate_all_descriptors()

        # Filter data for the specific protein
        protein_data = self.data[self.data['protein_name'] == protein_name]

        st.title(f"Molecular Descriptors Distribution Analysis")

        # Multi-select for choosing descriptors (default: all selected)
        selected_descriptors = st.multiselect("Select Descriptors", self.descriptors, default=self.descriptors, key = f'viz_desc_{protein_name}')

        # Create a grid of subplots: 2 rows (histograms and violin plots) and 5 columns
        num_descriptors = len(selected_descriptors)
        if num_descriptors > 0:
            fig, axes = plt.subplots(2, num_descriptors, figsize=(5 * num_descriptors, 10))
            fig.suptitle(f"Descriptor Distributions", fontsize=16)

            for i, desc in enumerate(selected_descriptors):
                # Histograms
                sns.histplot(
                    data=protein_data, x=f'mol_{desc}', hue='binds_plot', kde=True,
                    ax=axes[0, i], palette=self.palette
                )
                axes[0, i].set_title(f"Histogram: {desc.capitalize()}")
                axes[0, i].set_xlabel(desc.capitalize())
                axes[0, i].set_ylabel("Frequency")

                # Violin Plots
                sns.violinplot(
                    data=protein_data, x='binds_plot', y=f'mol_{desc}',
                    ax=axes[1, i], hue='binds_plot', palette=self.palette, legend=False
                )
                axes[1, i].set_title(f"Violin Plot: {desc.capitalize()}")
                axes[1, i].set_xlabel("Binds")
                axes[1, i].set_ylabel(desc.capitalize())

            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

        # === SCATTER PLOT MATRIX ===
        sns.set_style("whitegrid")
        st.write("### Scatter Plot Matrix")

        block_selected_descriptors = st.multiselect(
            "Select Descriptors for Scatter Plot Matrix",
            selected_descriptors, default=selected_descriptors,
            key=f"scatter_desc_{protein_name}"
        )

        if len(block_selected_descriptors) > 1:
            scatter_data = protein_data[[f'mol_{desc}' for desc in block_selected_descriptors] + ['binds_plot']].copy()
            scatter_data.columns = block_selected_descriptors + ['binds_plot']

            pairplot_fig = sns.pairplot(
                scatter_data,
                diag_kind="kde",
                hue='binds_plot',
                palette=self.palette,
                plot_kws={'alpha': 0.9}
            )
            st.pyplot(pairplot_fig.fig, clear_figure=True)


    def space_viz(self, protein_name):

        """
        Performs and visualizes dimensionality reduction using PCA (3D) and t-SNE (2D)

        Args:
            protein_name (str): Name of the protein
        """

        # === 3D PCA AND 2D T-SNE ===
        st.write("### Combined Feature Analysis: 3D PCA and 2D t-SNE")

        # Combine features for all descriptors
        combined_features = []
        for desc in self.descriptors:
            combined_features.append(f'mol_{desc}')

        # Extract combined features and drop rows with missing values
        protein_data = self.data[self.data['protein_name'] == protein_name]
        combined_data = protein_data[combined_features].dropna()

        # Perform PCA for 3D visualization
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(combined_data)
        pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
        pca_df['binds_plot'] = self.data.loc[combined_data.index, 'binds_plot']

        # Perform t-SNE for 2D visualization
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(combined_data)
        tsne_df = pd.DataFrame(tsne_result, columns=['t-SNE1', 't-SNE2'])
        tsne_df['binds_plot'] = self.data.loc[combined_data.index, 'binds_plot']

        # === 3D PCA Plot ===
        st.write("#### 3D PCA Plot")
        fig_3d = px.scatter_3d(
            pca_df, 
            x='PC1', 
            y='PC2', 
            z='PC3', 
            color='binds_plot',
            title="3D PCA Plot of Combined Features",
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
            color_discrete_sequence= ['#1f77b4', '#ff7f0e']
        )
        st.plotly_chart(fig_3d)

        # === 2D t-SNE Plot ===
        st.write("#### 2D t-SNE Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=tsne_df, x='t-SNE1', y='t-SNE2', hue='binds_plot',
            palette=['#1f77b4', '#ff7f0e'], ax=ax
        )
        ax.set_title("2D t-SNE Plot of Combined Features")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        st.pyplot(fig, clear_figure=True)
        


    def perform_statistical_tests(self, protein_name):
        """
        Performs statistical tests on molecular descriptors for a specific protein

        Args:
            protein_name (str): Name of the protein
        
        Returns:
            pd.DataFrame: Results of statistical tests
        """
        st.title(f"Statistical Tests")

        # Filter data for the specific protein
        protein_data = self.data[self.data['protein_name'] == protein_name]

        results = []

        # Perform statistical tests for each descriptor
        for desc in self.descriptors:
            col_name = f'mol_{desc}'

            # Extract data for binds and predictions
            bind_data = protein_data[protein_data['binds'] == 1][col_name].dropna()
            pred_1_data = protein_data[protein_data['Predicted Class'] == 1][col_name].dropna()
            ks_stat_binds, p_value_binds = ks_2samp(bind_data, pred_1_data)

            pred_0_data = protein_data[protein_data['Predicted Class'] == 0][col_name].dropna()
            no_bind_data = protein_data[protein_data['binds'] == 0][col_name].dropna()
            ks_stat_no_binds, p_value_no_binds = ks_2samp(pred_0_data, no_bind_data)

            # Interpret p-value
            def interpret_p_value(p_value):
                if p_value < 0.05:
                    return "Reject H0: Distributions are different"
                else:
                    return "Fail to reject H0: Distributions are similar"

            # Append results
            results.append({
                "Descriptor": desc,
                "KS Statistic (Binds)": np.round(ks_stat_binds, 2),
                "P-Value (Binds)": np.round(p_value_binds, 2),
                "Conclusion (Binds)": interpret_p_value(p_value_binds),
                "KS Statistic (No Binds)": np.round(ks_stat_no_binds, 2),
                "P-Value (No Binds)": np.round(p_value_no_binds, 2),
                "Conclusion (No Binds)": interpret_p_value(p_value_no_binds)
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Display results with filters
        with st.expander("### Show/Hide Full Results of Statistical Tests", expanded=False):
            st.write("### Filters")
            selected_descriptors = st.multiselect(
                "Select Descriptors to Display",
                self.descriptors,
                default=self.descriptors,
                key=f"desc_filter_{protein_name}"
            )

            # Filter results based on selected descriptors
            filtered_results = results_df[results_df['Descriptor'].isin(selected_descriptors)]

            # Display the filtered results
            st.table(filtered_results)

        st.write("""
        **Interpretation:**
        - **H0 (Null Hypothesis):** The distributions are the same.
        - **P-Value < 0.05:** Reject H0, indicating that the distributions are significantly different.
        - **P-Value >= 0.05:** Fail to reject H0, indicating that the distributions are similar.
        """)


    def visualize_top_molecules(self, protein):
        """
        Visualizes the top 4 molecules (by predicted probability) for each class

        Args:
            protein (str): Name of the protein
        """
        st.write(f"### Top Molecules for Protein: {protein}")

        # Load the data for the protein
        protein_data = self.data[self.data['protein_name'] == protein]

        # Separate class 0 and class 1
        class_0_data = protein_data[protein_data['Predicted Class'] == 0]
        class_1_data = protein_data[protein_data['Predicted Class'] == 1]

        # Sort class 0 by smallest predicted probability (non-binders)
        top_class_0 = class_0_data.nsmallest(4, 'Predicted Probability')

        # Sort class 1 by largest predicted probability (binders)
        top_class_1 = class_1_data.nlargest(4, 'Predicted Probability')

        # Function to visualize important blocks
        def visualize_important_blocks(molecule_smiles, bit_info):
            """
            Highlight important blocks in a molecule based on bit_info.
            """
            mol = Chem.MolFromSmiles(molecule_smiles)
            if mol is None:
                return None

            # Get the atoms contributing to the important bits
            important_atoms = set()
            for bit, (_, atoms) in bit_info.items():
                important_atoms.update(atoms)

            # Highlight important atoms in red
            highlight_atoms = list(important_atoms)
            img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms, highlightAtomColors={i: (1, 0, 0) for i in highlight_atoms})

            return img

        # Function to display molecules in a grid
        def display_molecules_grid(molecules_class_0, molecules_class_1):
            """
            Display molecules from class 0 and class 1 in a grid layout (4 per row)

            Args:
                molecules_class_0 (DataFrame): DataFrame containing molecules for Class 0
                molecules_class_1 (DataFrame): DataFrame containing molecules for Class 1
                bits_info (dict): Dictionary mapping molecule IDs to bit importance info
            """
            st.write("#### Top 4 Molecules for Class 0 (Non-Binders) and Class 1 (Binders)")
            cols = st.columns(4)  

            combined_molecules = list(molecules_class_0.iterrows()) + list(molecules_class_1.iterrows())

            for idx, (_, row) in enumerate(combined_molecules):
                molecule_smiles = row['molecule_smiles']
                molecule_id = row['id']
                bit_info = self.bits_info.get(molecule_id, {})

                # Visualize important blocks
                img = visualize_important_blocks(molecule_smiles, bit_info)
                if img:
                    with cols[idx % 4]:
                        st.image(img, caption=f"Molecule {molecule_id}", use_container_width=True)
                        st.write(f"**SMILES:** {molecule_smiles}")
                        st.write(f"**Probability:** {row['Predicted Probability']:.4f}")
                        st.write(f"**Class:** {'Binder' if row['Predicted Class'] == 1 else 'Non-Binder'}")
                else:
                    with cols[idx % 4]:
                        st.write(f"**Molecule {molecule_id}:** Unable to visualize.")

        # Display top molecules for class 0 and class 1 in one line
        display_molecules_grid(top_class_0, top_class_1)


    def run_visualizations(self):
        """
        Run all visualizations in Streamlit
        """
        st.title("Classification Model Visualization")

        self.threshold = st.sidebar.slider("Select Probability Threshold", 0.0, 1.0, 0.5, 0.1)
        self.y_pred = (self.y_pred_prob >= self.threshold).astype(int)

        self.plot_probability_distribution()
        self.plot_class_counts()
        if self.y_true is not None:
            self.plot_model_statistics()
            self.run_ab_test()
        def conditional_perform_statistical_tests(protein):
            if self.y_true is not None:
                self.perform_statistical_tests(protein)
            else:
                st.warning(f"Skipping statistical tests for {protein} because `y_true` is None.")

        analysis_functions = [
            self.visualize_distributions,
            conditional_perform_statistical_tests,
            self.space_viz,
            self.visualize_top_molecules,
        ]

        for protein in self.data['protein_name'].unique():
            st.header(f"Plots for {protein}")
            list(map(lambda func: func(protein), analysis_functions))
        