import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Nastavitve poti
train_dir = 'C:/Users/mitja/Desktop/podatki/train'
test_dir = 'C:/Users/mitja/Desktop/podatki/test'

def create_dataframe(directory):
     # Preverite, ali imenik obstaja
    if not os.path.isdir(directory):
        raise ValueError(f"The directory does not exist: {directory}")

    filepaths = []
    labels = []
    # Če imenik vsebuje podimenike, ki predstavljajo kategorije
    if any(os.path.isdir(os.path.join(directory, i)) for i in os.listdir(directory)):
        categories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        for category in categories:
            cat_dir = os.path.join(directory, category)
            files = os.listdir(cat_dir)
            for file in files:
                filepaths.append(os.path.join(cat_dir, file))
                labels.append(category)
    else:  # Če imenik neposredno vsebuje datoteke
        for file in os.listdir(directory):
            if file.endswith('.jpg') or file.endswith('.png'):  # Preverite, ali je datoteka slika
                filepaths.append(os.path.join(directory, file))
                labels.append(None)  # Ni kategorije za testne slike
    return pd.DataFrame({'filepath': filepaths, 'label': labels})

def build_cnn_model(input_shape, num_classes, filters=16, dense_neurons=128):
    model = Sequential([
        Conv2D(filters, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(filters, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(dense_neurons, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, validation_generator, epochs=100):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )
    return history

def plot_learning_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

# Dodana funkcija za primerjavo modelov
def compare_models(histories):
    val_accuracies = [max(history['val_accuracy']) for history in histories]
    best_model_index = np.argmax(val_accuracies)
    return best_model_index

def evaluate_model(model, test_generator):
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
    return test_loss, test_accuracy

def save_predictions(model, test_generator, file_name='submission.csv'):
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    filenames = test_generator.filenames
    results = pd.DataFrame({"Filename": filenames, "Predictions": predicted_classes})
    results.to_csv(file_name, index=False)

def main():
    train_df = create_dataframe(train_dir)
    test_df = create_dataframe(test_dir)

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1./255)

    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    fold_var = 1
    histories = []

    for train_index, val_index in kf.split(train_df):
        training_data = train_df.iloc[train_index]
        validation_data = train_df.iloc[val_index]

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=training_data,
            x_col='filepath',
            y_col='label',
            target_size=(100, 100),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )

        validation_generator = train_datagen.flow_from_dataframe(
            dataframe=validation_data,
            x_col='filepath',
            y_col='label',
            target_size=(100, 100),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            shuffle=True,
            seed=42
        )

        # Spremenljivi hiperparametri
        filters = 16 if fold_var <= 3 else 32  # Spremenite število filtrov
        dense_neurons = 128 if fold_var <= 3 else 256  # Spremenite število nevronov v gostem sloju

        model = build_cnn_model((100, 100, 1), 3, filters=filters, dense_neurons=dense_neurons)
        history = train_model(model, train_generator, validation_generator, epochs=100)

        plot_learning_curves(history)

        histories.append(history.history)

        model.save(f'cnn_model_fold_{fold_var}.h5')

        fold_var += 1

    # Primerjava modelov in izbira najboljšega
    best_model_index = compare_models(histories)
    print(f"Najboljši model je model št. {best_model_index+1}")

    # Ustvarjanje iteratorjev slik za testno množico
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepath',
        target_size=(100, 100),
        color_mode='grayscale',
        batch_size=1,
        class_mode=None,
        shuffle=False
    )

     # Nalaganje in uporaba najboljšega modela
    best_model = load_model(f'cnn_model_fold_{best_model_index+1}.h5')
    evaluate_model(best_model, test_generator)

    # Shranjevanje napovedi za Kaggle
    save_predictions(best_model, test_generator, file_name='kaggle_submission.csv')

if __name__ == "__main__":
    main()