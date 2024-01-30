import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Izgradnja seznama poti do slik podatkovne množice
data_dir = 'C:/Users/mitja/Desktop/shapes'
categories = ['circles', 'squares', 'triangles']
paths = []
labels = []

for i, category in enumerate(categories):
    category_dir = os.path.join(data_dir, category)
    for filename in os.listdir(category_dir):
        paths.append(os.path.join(category_dir, filename))
        labels.append(i)

# Izris primerka slike posamezne kategorije
for i, category in enumerate(categories):
    img_path = paths[labels.index(i)]
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(category)
    plt.show()

# Izgradnja dvodimenzionalnega polja slik
images = []
for path in paths:
    img = Image.open(path).convert('L')
    img_array = np.array(img).flatten()
    images.append(img_array)

X = np.array(images)
y = np.array(labels)

# Razdelitev podatkov
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4321, shuffle=True)

# Gradnja napovednega modela
clf = DecisionTreeClassifier(random_state=1234)
clf.fit(X_train, y_train)

# Izris odločitvenega drevesa
plot_tree(clf)
plt.show()

# Ovrednotenje napovednega modela
y_pred = clf.predict(X_test)
print(f'Točnost: {accuracy_score(y_test, y_pred)}')
print(f'Utežena F1 vrednost: {f1_score(y_test, y_pred, average="weighted")}')
print(f'Utežena preciznost: {precision_score(y_test, y_pred, average="weighted")}')
print(f'Utežen priklic: {recall_score(y_test, y_pred, average="weighted")}')




# Gradnja napovednega modela z algoritmom k-najbližjih sosedov
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Ovrednotenje napovednega modela
y_pred_knn = knn.predict(X_test)
print(f'Točnost k-najbližjih sosedov: {accuracy_score(y_test, y_pred_knn)}')

# Primerjava rezultatov obeh algoritmov v obliki grafov
plt.figure()
plt.bar(['Odločitveno drevo', 'K-najbližjih sosedov'],
        [accuracy_score(y_test, y_pred), accuracy_score(y_test, y_pred_knn)])
plt.ylabel('Točnost')
plt.show()
