from functionality.prediction import Predictor


class EnsembleModel:
    """
    A wrapper class for making ensemble predictions with bit information
    """
    def __init__(self):
        """
        Initialize the ensemble model with a Predictor instance
        """
        self.predictor = Predictor()

    def predict(self, model_input):
        """
        Make predictions on input data and return results with bit information
        
        Args:
            model_input: Input data for the predictor
            
        Returns:
            dict: Dictionary containing two keys:
                - "results": List of prediction results as dictionaries
                - "bit_info": Corresponding bit information for the predictions
        """
        results, bit_info = self.predictor.prediction_and_bit_info(model_input)
        results_dict = results.to_dict(orient='records')
        
        output = {
            "results": results_dict,
            "bit_info": bit_info
        }
        
        return output
