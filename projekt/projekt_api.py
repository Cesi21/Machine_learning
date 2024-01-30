from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

app = Flask(__name__)

model = load_model('lstm_model.h5')
scaler = joblib.load('scaler_pm10.pkl')

window_size = 64

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pm10_values = data['PM10']

    if len(pm10_values) < window_size:
        return jsonify({'error': 'Ni dovolj podatkov za ustvarjanje sekvence.'}), 400

    df = pd.DataFrame(pm10_values, columns=['PM10'])
    scaled_data = scaler.transform(df)

    input_sequence = np.array([scaled_data[-window_size:]])
    input_sequence = input_sequence.reshape(-1, 1, window_size)

    prediction = model.predict(input_sequence)
    original_scale_prediction = scaler.inverse_transform(prediction)
    return jsonify({'prediction': original_scale_prediction.flatten().tolist()})

if __name__ == '__main__':
    app.run(debug=True)
