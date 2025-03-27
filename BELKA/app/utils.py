import functools
import json
import pandas as pd
import boto3
from rdkit import Chem 

sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Validation function
def validate_input_data(data):
    """
    Validates the input dataset for correctness

    Parameters:
    - data (pandas.DataFrame): The input dataset containing 'protein_name' and 'molecule_smiles'

    Returns:
    - True if the input data is valid

    Raises:
    - ValueError: If 'protein_name' contains invalid values or 'molecule_smiles' is missing/invalid
    """
    def is_valid_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    valid_proteins = ['HSA', 'BRD4', 'sEH']
    if 'protein_name' not in data.columns or not data['protein_name'].isin(valid_proteins).all():
        raise ValueError("Column 'protein_name' must contain only values from ['HSA', 'BRD4', 'sEH']")
    
    if 'molecule_smiles' not in data.columns or data['molecule_smiles'].isnull().any():
        raise ValueError("Column 'molecule_smiles' must be present and cannot contain NaN values")
    
    invalid_smiles = data[~data['molecule_smiles'].apply(is_valid_smiles)]
    if not invalid_smiles.empty:
        raise ValueError(f"Invalid SMILES strings found: {invalid_smiles['molecule_smiles'].tolist()}")
    
    return True


# Decorator for validation
def validate_input_decorator(func):
    """
    A decorator to validate input data before executing the wrapped function

    Parameters:
    - func (callable): The function to be wrapped

    Returns:
    - Callable: Wrapped function with input validation

    Raises:
    - ValueError: If input data validation fails
    """
    @functools.wraps(func)
    def wrapper(data):
        try:
            validate_input_data(data)
            return func(data)
        except ValueError as e:
            raise ValueError(f"Input data validation failed: {e}")
    return wrapper

# Decorator for invoking the endpoint
def invoke_endpoint_decorator(func):
    """
    A decorator to send input data to a SageMaker endpoint and process the response

    Parameters:
    - func (callable): The function to be wrapped

    Returns:
    - Callable: Wrapped function with SageMaker invocation

    Raises:
    - Exception: If the endpoint invocation fails
    """
    @functools.wraps(func)
    def wrapper(data):
        try:
            # Validate input data
            validate_input_data(data)
            
            # Prepare payload
            payload = data.to_json(orient='split')
            
            # Invoke the SageMaker endpoint
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName='ensemble-model-2025-03-24-10-05-48-038',
                ContentType='application/json',
                Body=payload
            )
            
            # Parse the response
            result = response['Body'].read().decode('utf-8')
            result_data = json.loads(result)
            
            # Extract results and bit_info
            results_df = pd.DataFrame(result_data['results'])
            bit_info = result_data['bit_info']
            
            # Call the original function with the results
            return func(results_df, bit_info)
        
        except Exception as e:
            raise Exception(f"Failed to invoke endpoint: {e}")
    
    return wrapper