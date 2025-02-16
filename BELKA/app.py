import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.modules.pop("functionality.prediction", None)
sys.modules.pop("functionality.analysis", None)
from functionality.prediction import Predictor
from functionality.analysis import Analyser


st.title('Prediction of binding ligands')

uploaded_file = st.file_uploader('Load file with molecules in .csv or .parquet format', type = ['csv', 'parquet'])

if uploaded_file is not None:

    if Path(uploaded_file.name).suffix == '.csv':
        test = pd.read_csv(uploaded_file)
    elif Path(uploaded_file.name).suffix == '.parquet':
        test = pd.read_parquet(uploaded_file)
    else:
        raise TypeError('Expected file in .csv or .parquet format')
    

    predictor = Predictor(test.drop('binds', axis = 1))
    y_pred, y_pred_prob, data, bit_infos, lgb_cls = predictor.predict()

    data['Predicted Class'] = y_pred
    data['Predicted Probability'] = y_pred_prob[:, 1]
    data['binds'] = test['binds']
    data['buildingblock1_smiles'] = test['buildingblock1_smiles']
    data['buildingblock2_smiles'] = test['buildingblock2_smiles']
    data['buildingblock3_smiles'] = test['buildingblock3_smiles']
    data['molecule_smiles'] = test['molecule_smiles']

    st.write('Prediction Results: ')

    st.write(pd.DataFrame({
        "Predicted class": y_pred,
        "Binding Probability": y_pred_prob[:, 1]
    }))


    analyzer = Analyser(data, bit_infos, lgb_cls)
    analyzer.run_visualizations()


