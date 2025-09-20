# server.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('regression_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()
        # Expecting {"x1": value1, "x2": value2}
        X = np.array([[data['x1'], data['x2']]])
        prediction = model.predict(X)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

