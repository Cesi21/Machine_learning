# Uvoz potrebnih knjižnic
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Obdelava podatkov za klasifikacijski problem
def load_and_preprocess_classification_data():
    data_dir = 'C:/Users/mitja/Desktop/shapes'
    categories = ['circles', 'squares', 'triangles']
    paths = []
    labels = []

    # Zbiranje poti in oznak slik
    for i, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        for filename in os.listdir(category_dir):
            paths.append(os.path.join(category_dir, filename))
            labels.append(i)

    # Pretvorba slik v polje
    images = []
    for path in paths:
        img = Image.open(path).convert('L')
        img_array = np.array(img).flatten()
        images.append(img_array)

    X = np.array(images)
    y = np.array(labels)

    return X, y

# Obdelava podatkov za regresijski problem
def load_and_preprocess_regression_data():
    # Nalaganje podatkov
    data = pd.read_csv('C:/Users/mitja/Desktop/bike_data.csv')

    # Zapolnitev manjkajočih podatkov
    for column in data.columns:
        if data[column].dtype in [np.float64, np.int64]:
            data[column].fillna(data[column].mean(), inplace=True)

    # Pretvorba stolpca "date"
    data['day'] = data['date'].str.split('/').str[0].astype(int)
    data['month'] = data['date'].str.split('/').str[1].astype(int)
    data['year'] = data['date'].str.split('/').str[2].astype(int)
    data.drop('date', axis=1, inplace=True)

    # Kodiranje kategoričnih spremenljivk
    data = pd.get_dummies(data, columns=['seasons', 'holiday', 'work_hours'])

    return data

# Ansambelski modeli
def build_regressor_models():
    models = {
        "Bagging": BaggingRegressor(random_state=1234, n_jobs=-1),
        "Random Forest": RandomForestRegressor(random_state=1234, n_jobs=-1),
        "AdaBoost": AdaBoostRegressor(random_state=1234),
        "Gradient Boosting": GradientBoostingRegressor(random_state=1234)
    }
    return models

def build_classifier_models():
    models = {
        "Bagging": BaggingClassifier(random_state=1234, n_jobs=-1),
        "Random Forest": RandomForestClassifier(random_state=1234, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(random_state=1234),
        "Gradient Boosting": GradientBoostingClassifier(random_state=1234)
    }
    return models

# 5-kratna prečna validacija in ovrednotenje modelov
def evaluate_models(models, X, y, problem_type="regression"):
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    results = {}

    for name, model in models.items():
        if problem_type == "regression":
            metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        else:
            metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        
        cv_results = cross_validate(model, X, y, cv=kf, scoring=metrics)
        results[name] = cv_results

    return results

# Vizualizacija rezultatov
def plot_results(results, problem_type="regression"):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_scores = []
    names = []
    metrics = []

    for name, scores in results.items():
        for metric, values in scores.items():
            if 'time' not in metric:
                all_scores.extend(values)
                names.extend([name] * len(values))
                metrics.extend([metric] * len(values))

    data_to_plot = pd.DataFrame({'Model': names, 'Score': all_scores, 'Metric': metrics})
    sns.boxplot(x='Model', y='Score', hue='Metric', data=data_to_plot, ax=ax)

    ax.set_title(f'Performance Comparison ({problem_type.capitalize()} Problem)')
    ax.set_ylabel('Score')
    plt.legend(title='Metric')
    plt.show()

# Vizualizacija povprečnih vrednosti metrik
def plot_average_scores(results, problem_type="regression"):
    avg_scores = {}
    for name, scores in results.items():
        avg_scores[name] = {metric: np.mean(values) for metric, values in scores.items() if 'time' not in metric}

    df_avg_scores = pd.DataFrame(avg_scores)
    df_avg_scores.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Average Scores ({problem_type.capitalize()} Problem)')
    plt.ylabel('Score')
    plt.show()

# Glavni program
def main():
    # Obdelava podatkov
    regression_data = load_and_preprocess_regression_data()
    X_reg = regression_data.drop('rented_bike_count', axis=1)
    y_reg = regression_data['rented_bike_count']

    classification_data = load_and_preprocess_classification_data()
    X_cls, y_cls = classification_data

    # Gradnja modelov
    regressor_models = build_regressor_models()
    classifier_models = build_classifier_models()

    # Evaluacija modelov
    regression_results = evaluate_models(regressor_models, X_reg, y_reg, problem_type="regression")
    classification_results = evaluate_models(classifier_models, X_cls, y_cls, problem_type="classification")

    # Vizualizacija rezultatov
    plot_results(regression_results, problem_type="regression")
    plot_results(classification_results, problem_type="classification")

    # Vizualizacija povprečnih vrednosti metrik
    plot_average_scores(regression_results, problem_type="regression")
    plot_average_scores(classification_results, problem_type="classification")

if __name__ == "__main__":
    main()
