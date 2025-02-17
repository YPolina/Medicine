import pickle 

import pandas as pd 
import numpy as np

from rdkit.Chem import MolFromSmiles, DataStructs, rdFingerprintGenerator
from rdkit import Chem

def load_model():
    with open("./lgb_cls.pkl", "rb") as f:
        model = pickle.load(f)
    return model

lgb_cls = load_model()





class Predictor:

    def __init__(self, data: pd.DataFrame):

        "Class initialization"

        self.data = data.copy()
        self.fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024, useBondTypes=True, includeChirality=True, includeRingMembership=True)

    def data_validation(self):
        """
        Input check for format corresponding

        """
        #Check for nessesary columns
        required_columns = ['id', 'buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles', 'protein_name']
        
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f'Column {col} is missing in input data')
            

        #Validate protein name
        valid_proteins = {'BRD4', 'sEH', 'HSA'}

        if not self.data['protein_name'].isin(valid_proteins).all():
            raise ValueError(f"Protein name must be one of {valid_proteins}")

            
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
            return np.zeros(1024, dtype=np.int8)
        
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        
        
        fp = self.fpg.GetCountFingerprint(mol, additionalOutput=ao)

        bit_info = ao.GetBitInfoMap()
        arr = np.zeros(1024, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
    
        return arr, bit_info
    
    def feature_engineering(self, smiles_columns = ['molecule_smiles']):
        """
        Computes fingerprints for specified SMILES columns, concatenates them, and prepares feature matrix (X)

        Args:
            
            smiles_columns (list): List of column names containing SMILES strings

        Returns:
            X (pd.DataFrame): Feature matrix containing concatenated fingerprints
        """

        results = self.data['molecule_smiles'].apply(self.compute_fn)
        self.data['fp'] = results.apply(lambda x: x[0])
        bit_infos = results.apply(lambda x: x[1]).tolist()

        protein_dummies = pd.get_dummies(self.data['protein_name'], prefix='protein', dtype = 'int')

        fingerprint_df = pd.DataFrame(self.data["fp"].to_list(), index=self.data.index)

        self.data = pd.concat([self.data.drop(columns=["fp"]), fingerprint_df, protein_dummies], axis=1)

        return bit_infos

    
    def predict(self):
        self.data_validation()

        #Data transformation
        bit_infos = self.feature_engineering()

        columns = ['id', 'molecule_smiles', 'protein_name', 'buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'binds']

        features = [feature for feature in self.data.columns if feature not in columns]
        X = self.data[features]

        y_pred = lgb_cls.predict(X)
        y_pred_prob = lgb_cls.predict_proba(X)

        self.data['Predicted Class'] = y_pred
        self.data['Predicted Probability'] = y_pred_prob[:, 1]

        return self.data, bit_infos, lgb_cls
