import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Naložimo podatke
data = pd.read_csv('mbajk.csv')

# Sortiramo podatke po času
data['date'] = pd.to_datetime(data['date'])
data.sort_values(by='date', inplace=True)
data.set_index('date', inplace=True)

# Izberemo ciljno značilnost 'available_bike_stands'
data = data[['available_bike_stands']]

# Pretvorimo podatke v numpy array
data_array = data.values.astype(float)

# Standardiziramo podatke
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_array)

# Razdelimo podatke na učno in testno množico
train_size = len(data_scaled) - 1302
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

# Funkcija za pretvorbo časovne vrste v obliko primerno za RNN
def create_dataset(dataset, window_size):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:i+window_size])
        y.append(dataset[i+window_size])
    return np.array(X), np.array(y)

# Nastavimo velikost okna
window_size = 186

# Ustvarimo učne in testne podatke
X_train, y_train = create_dataset(train_data, window_size)
X_test, y_test = create_dataset(test_data, window_size)

# Preoblikujemo podatke za RNN
X_train = X_train.reshape(-1, 1, window_size)
X_test = X_test.reshape(-1, 1, window_size)

# Funkcija za izgradnjo modela
def build_model():
    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(32, return_sequences=True, input_shape=(1, window_size)))
    model.add(keras.layers.SimpleRNN(32))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1))
    return model

# Ustvarimo in kompiliramo model
model = build_model()
model.compile(optimizer='adam', loss='mse')

# Učenje modela
epochs = 50
history = model.fit(X_train, y_train, epochs=epochs, verbose=1)

# Funkcija za obrnjeno standardizacijo
def inverse_transform(scaler, data):
    return scaler.inverse_transform(data.reshape(-1, 1))

# Izračunamo napovedi
predictions_test = model.predict(X_test)

# Obrnemo standardizacijo
y_train_original = inverse_transform(scaler, y_train)
y_test_original = inverse_transform(scaler, y_test)
predictions_test_original = inverse_transform(scaler, predictions_test)

# Pridobimo datume za učne in testne vzorce
train_dates = data.index[:len(y_train_original)]
test_dates = data.index[len(y_train_original):len(y_train_original) + len(y_test_original)]

# Izris grafov
plt.figure(figsize=(15, 6))

# Graf 1: Dejanske vrednosti za učno množico in napovedi za testno množico
plt.subplot(1, 2, 1)
plt.plot(train_dates, y_train_original, label='Actual Training Data')
plt.plot(test_dates, predictions_test_original, label='Predicted Test Data')
plt.plot(test_dates, y_test_original, label='Actual Test Data')
plt.title('Training and Test Data')
plt.xlabel('Date')
plt.ylabel('Available Bike Stands')
plt.legend()

# Graf 2: Napovedi in dejanske vrednosti za testno množico
plt.subplot(1, 2, 2)
plt.plot(test_dates, y_test_original, label='Actual')
plt.plot(test_dates, predictions_test_original, label='Predicted')
plt.title('Prediction vs Actual for Test Data')
plt.xlabel('Date')
plt.ylabel('Available Bike Stands')
plt.legend()

plt.tight_layout()
plt.show()