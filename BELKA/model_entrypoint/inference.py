from flask import Flask, request, jsonify
import pandas as pd
from functionality.ensemble_model import EnsembleModel

app = Flask(__name__)

model = EnsembleModel()

@app.route("/ping", methods=["GET"])
def ping():
    """
    Health check endpoint

    Returns:
    - Empty response with HTTP status 200 if the service is running
    """
    return "", 200

@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Handles prediction requests

    Expects:
    - JSON payload with:
      - 'data': A list of molecular input data
      - 'columns': Column names corresponding to the data

    Returns:
    - JSON response containing model predictions
    """
    input_data = request.get_json()
    model_input = pd.DataFrame(input_data['data'], columns=input_data['columns'])
    results = model.predict(model_input)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)