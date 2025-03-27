import streamlit as st
import pandas as pd
from pathlib import Path
from utils import invoke_endpoint_decorator
import boto3
from analysis import Analyser

@invoke_endpoint_decorator
def get_predictions(results_df, bit_info):
    return results_df, bit_info


# Initialize the SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Streamlit app title
st.title('Prediction of binding ligands with protein sEH')

# File uploader
uploaded_file = st.file_uploader('Load file with molecules in .csv or .parquet format', type=['csv', 'parquet'])

if uploaded_file is not None:
    # Read the uploaded file
    if Path(uploaded_file.name).suffix == '.csv':
        data = pd.read_csv(uploaded_file)
    elif Path(uploaded_file.name).suffix == '.parquet':
        data = pd.read_parquet(uploaded_file)
    else:
        raise TypeError('Expected file in .csv or .parquet format')
    
    # Validate the input data
    try:
         results_df, bit_info = get_predictions(data[:300])
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    
    # Display the prediction results
    st.write('Prediction Results:')
    st.write(pd.DataFrame({
        "Predicted class": results_df['Predicted Class'],
        "Binding Probability": results_df['Predicted Probability']
    }))

    # Run visualizations
    analyzer = Analyser(results_df, bit_info)
    analyzer.run_visualizations()