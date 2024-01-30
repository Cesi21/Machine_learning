
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.metrics import F1Score
from keras.models import load_model
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import io
import base64
import joblib
import pandas as pd

app = Flask(__name__)

#model_naloga4 = joblib.load('C:/Users/mitja/Desktop/Strojno/tretji zagovor/random_forest_model.pkl')
#model_naloga6 = joblib.load('C:/Users/mitja/Desktop/Strojno/tretji zagovor/best_model_GRU.pkl')
#model = load_model('C:/Users/mitja/Desktop/Strojno/tretji zagovor/best_cnn_model.h5')
#scaler4 = joblib.load('C:/Users/mitja/Desktop/Strojno/tretji zagovor/scaler4.pkl')
#descaler4 = joblib.load('C:/Users/mitja/Desktop/Strojno/tretji zagovor/target_scaler.pkl')
#scaler6 = joblib.load('C:/Users/mitja/Desktop/Strojno/tretji zagovor/scaler6.pkl')

model_naloga4 = joblib.load('random_forest_model.pkl')
model_naloga6 = load_model('best_model_GRU.h5')
model = load_model('best_cnn_model.h5')
scaler4 = joblib.load('scaler4.pkl')
descaler4 = joblib.load('target_scaler.pkl')
scaler6 = joblib.load('scaler6.pkl')

def preprocess_input_naloga4(data):

    df = pd.DataFrame([data])
    df.rename(columns={'date': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['hour'] = df['timestamp'].dt.hour
    df.drop('timestamp', axis=1, inplace=True)
    df['index'] = 0
    #print(df.head)
    column_order = ['index', 'temperature', 'relative_humidity', 'dew_point', 
                'apparent_temperature', 'precipitation_probability', 'rain', 
                'surface_pressure', 'bike_stands', 'day', 'month', 'year', 'hour']

    df = df[column_order]
    dataB = scaler4.transform(df)
    #print(dataB)
    return dataB



@app.route('/napoved/naloga4', methods=['POST'])
def napoved_naloga4():
    data = request.get_json()
    input_data = preprocess_input_naloga4(data)
    prediction = model_naloga4.predict(input_data)
    rez = descaler4.inverse_transform(prediction.reshape(-1, 1)).flatten()
    return jsonify({'prediction': rez[0]})


    
@app.route('/napoved/naloga6', methods=['POST'])
def napoved_naloga6():
    data = request.get_json()
    timeseries = data.get('timeseries')

    if len(timeseries) < 186:
        return jsonify({'error': 'Invalid timeseries length (too short). Expected 186 data points'}), 400
    if len(timeseries) > 186:
        return jsonify({'error': 'Invalid timeseries length (too long). Expected 186 data points'}), 400

    timeseries_reshaped = np.array([timeseries]).reshape(1, 1, 186)
    prediction = model_naloga6.predict(timeseries_reshaped)
    prediction_descaled = scaler6.inverse_transform(prediction.reshape(-1, 1))
    return jsonify({'prediction': prediction_descaled.tolist()})

@app.route('/napoved/naloga5', methods=['POST'])
def napoved_naloga5():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    image = image.resize((100, 100)).convert('L') 
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)


    prediction = model.predict(image)
    class_names = ['paper', 'rock', 'scissors']
    predicted_class = class_names[np.argmax(prediction[0])]
    
    return jsonify({'prediction': predicted_class})



if __name__ == '__main__':
    app.run(debug=True)