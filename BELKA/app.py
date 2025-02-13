import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.modules.pop("functionality.prediction", None)
sys.modules.pop("functionality.analysis", None)
from functionality.prediction import Predictor
from functionality.analysis import Analyser


st.title('Prediction of binding ligands with protein sEH')

uploaded_file = st.file_uploader('Load file with molecules in .csv or .parquet format', type = ['csv', 'parquet'])

if uploaded_file is not None:

    if Path(uploaded_file.name).suffix == '.csv':
        data = pd.read_csv(uploaded_file)
    elif Path(uploaded_file.name).suffix == '.parquet':
        data = pd.read_parquet(uploaded_file)
    else:
        raise TypeError('Expected file in .csv or .parquet format')
    

    predictor = Predictor(data.drop('binds', axis = 1))
    y_pred, y_pred_prob, data, bit_infos = predictor.predict()

    data['Predicted Class'] = y_pred
    data['Predicted Probability'] = y_pred_prob[:, 1]

    st.write('Prediction Results: ')

    st.write(pd.DataFrame({
        "Predicted class": y_pred,
        "Binding Probability": y_pred_prob[:, 1]
    }))


    analyzer = Analyser(data, bit_infos)
    analyzer.run_visualizations()


