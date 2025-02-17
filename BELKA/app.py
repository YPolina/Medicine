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
    

    predictor = Predictor(data)
    data, bit_infos, model = predictor.predict()


    st.write('Prediction Results: ')

    st.write(pd.DataFrame({
        "Predicted class": data['Predicted Class'],
        "Binding Probability": data['Predicted Probability']
    }))


    analyzer = Analyser(data, bit_infos, model)
    analyzer.run_visualizations()


