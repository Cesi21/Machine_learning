# Uvoz potrebnih knjižnic
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import glob

# Naložimo podatke
data = pd.read_csv('C:\Users\mitja\Desktop\mbajk_dataset.csv', parse_dates=['date'], index_col='date')

# Osnovna obdelava podatkov za regresijski problem
# Napolnimo manjkajoče vrednosti
imputer = SimpleImputer(strategy='mean')
data.iloc[:,1:] = imputer.fit_transform(data.iloc[:,1:])

# Linear regression za napovedovanje manjkajočih vrednosti (če je potrebno)
# Primer: napolnite manjkajoče vrednosti za 'temperature'
if data['temperature'].isnull().any():
    reg = LinearRegression()
    not_null_data = data.dropna(subset=['temperature'])
    reg.fit(not_null_data.drop('temperature', axis=1), not_null_data['temperature'])
    data.loc[data['temperature'].isnull(), 'temperature'] = reg.predict(data[data['temperature'].isnull()].drop('temperature', axis=1))

# Izvedemo transformacijo vrednosti za numerične značilnice
pt = PowerTransformer()
data[['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature']] = pt.fit_transform(data[['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature']])

# Normalizacija ali standardizacija vrednosti
scaler = StandardScaler()
data[['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature']] = scaler.fit_transform(data[['temperature', 'relative_humidity', 'dew_point', 'apparent_temperature']])

# Zgradimo tri nove smiselne značilnice
data['day'] = data.index.day
data['month'] = data.index.month
data['hour'] = data.index.hour

# Izvedemo proces izbire značilnic s filtriranjem
X = data.drop('available_bike_stands', axis=1)  # Privzamemo, da je 'available_bike_stands' naša ciljna spremenljivka
y = data['available_bike_stands']
mi = mutual_info_regression(X, y)
mi /= np.max(mi)  # Normaliziramo vrednosti information gain na območje [0, 1]

# Izberemo značilnice z največjim information gain
selected_features = X.columns[mi > 0.1]  # Threshold je primeren za izbiro

# Obdelava podatkov za klasifikacijski problem
# Normalizacija polj slik
data_dir = 'C:\Users\mitja\Desktop\agriculture-crops'
categories = [
    "almond",
    "banana",
    "cardamom",
    "cherry",
    "chilli",
    "clove",
    "coconut",
    "coffee_plant",
    "cotton",
    "cucumber",
    "fox_nut-makhana",
    "gram",
    "jowar",
    "jute",
    "lemon",
    "maize",
    "mustard_oil",
    "olive_tree",
    "papaya",
    "pearl_millet-bajra",
    "pineapple",
    "rice",
    "soyabean",
    "sugarcane",
    "sunflower",
    "tea",
    "tobacco_plant",
    "tomato",
    "vigna_radiati-mung",
    "wheat"
]
paths = []
labels = []

for i, category in enumerate(categories):
    for filename in glob.glob(f'{data_dir}/{category}/*.jpg'):
        img = Image.open(filename).convert('L')
        img = img.resize((128, 128))  # Resize images for uniformity
        img_array = np.array(img).flatten()
        paths.append(img_array)
        labels.append(i)

X_images = np.array(paths)
y_images = np.array(labels)
X_images = X_images / 255.0  # Normaliziramo vrednosti slik na območje [0, 1]

# Izgradnja napovednega modela z uporabo nevronskih mrež
# Za regresijski problem
mlp_reg = MLPRegressor(random_state=1234)
# Za klasifikacijski problem
mlp_clf = MLPClassifier(random_state=1234)

# 5-kratna prečna validacija in ovrednotenje modelov
kf = KFold(n_splits=5, shuffle=True, random_state=1234)
regression_scores = cross_validate(mlp_reg, X[selected_features], y, cv=kf, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'], n_jobs=-1)
classification_scores = cross_validate(mlp_clf, X_images, y_images, cv=kf, scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'], n_jobs=-1)


# Vizualizacija rezultatov
def plot_results(results, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in results:
        if 'test' in metric and 'time' not in metric:
            data = -results[metric] if 'neg_' in metric else results[metric]
            sns.boxplot(data=data, ax=ax, label=metric.replace('test_neg_', '').replace('test_', ''))
    plt.title(title)
    plt.ylabel('Score')
    plt.legend()
    plt.show()

# Vizualizacija povprečnih vrednosti metrik
def plot_average_scores(results, title):
    averages = {metric: np.mean(scores) for metric, scores in results.items() if 'test' in metric and 'time' not in metric}
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(averages)), list(averages.values()), align='center')
    plt.xticks(range(len(averages)), list(averages.keys()), rotation=45)
    plt.title(title)
    plt.ylabel('Score')
    plt.show()

# Prikaz regresijskih rezultatov
plot_results(regression_scores, 'Regression Metrics')
plot_average_scores(regression_scores, 'Average Regression Metrics')

# Prikaz klasifikacijskih rezultatov
plot_results(classification_scores, 'Classification Metrics')
plot_average_scores(classification_scores, 'Average Classification Metrics')