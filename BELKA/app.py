import streamlit as st
import pandas as pd
from pathlib import Path

import functions
import importlib
importlib.reload(functions)
from functions import *


st.title('Prediction of binding ligands with protein sEH')

uploaded_file = st.file_uploader('Load file with molecules in .csv or .parquet format', type = ['csv', 'parquet'])

if uploaded_file is not None:

    if Path(uploaded_file.name).suffix == '.csv':
        data = pd.read_csv(uploaded_file)
    elif Path(uploaded_file.name).suffix == '.parquet':
        data = pd.read_parquet(uploaded_file)
    else:
        raise TypeError('Expected file in .csv or .parquet format')
    

    predictor = Prediction(data)

    y_pred, y_pred_proba, processed_data = predictor.predict()

    st.write('Prediction Results: ')

    st.write(pd.DataFrame({
        "Predicted class": y_pred,
        "Binding Probability": y_pred_proba
    }))


    analysis = Analysis(processed_data, y_pred)
    analysis.run_analysis(output='streamlit')


