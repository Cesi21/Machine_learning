import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import csv

from sklearn import datasets
from sklearn.model_selection import train_test_split , KFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from collections import Counter



iris = datasets.load_iris()

iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                      columns= iris['feature_names'] + ['target'])
iris_df.head()
iris_df.describe()
x= iris_df.iloc[:, :-1]
y= iris_df.iloc[:, -1]
x.head()
y.head()

i = input('vnesite E ali M za način računanja')

x_train, x_test, y_train, y_test= train_test_split(x, y,
                                                   test_size= 0.2,
                                                   shuffle= True,
                                                   random_state= 0)
x_train= np.asarray(x_train)
y_train= np.asarray(y_train)

x_test= np.asarray(x_test)
y_test= np.asarray(y_test)


print(f'training set size: {x_train.shape[0]} samples \ntest set size: {x_test.shape[0]} samples')



scaler= Normalizer().fit(x_train) 
normalized_x_train= scaler.transform(x_train)
normalized_x_test= scaler.transform(x_test)


#normalized_x_train= x_train 
#normalized_x_test= x_test

def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

def distance_ecu(x_train, x_test_point):
 
  distances= []  
  for row in range(len(x_train)): 
      current_train_point= x_train[row] 
      current_distance= 0 

      for col in range(len(current_train_point)): 
          
          current_distance += (current_train_point[col] - x_test_point[col]) **2
        
      current_distance= np.sqrt(current_distance)

      distances.append(current_distance) 

  
  distances= pd.DataFrame(data=distances,columns=['dist'])
  return distances

def distance_M(x_train, x_test_point):
 
  distances= []  
  for row in range(len(x_train)): 
      current_train_point= x_train[row] 
      current_distance= 0 

      for col in range(len(current_train_point)): 
          
          current_distance += abs(current_train_point[col]-x_test_point[col])
        
      

      distances.append(current_distance) 

  
  distances= pd.DataFrame(data=distances,columns=['dist'])
  return distances

def nearest_neighbors(distance_point, K):
  
    df_nearest= distance_point.sort_values(by=['dist'], axis=0)

   
    df_nearest= df_nearest[:K]
    return df_nearest



def voting(df_nearest, y_train):
    
    counter_vote= Counter(y_train[df_nearest.index])

    y_pred= counter_vote.most_common()[0][0]   

    return y_pred



def KNN_from_scratch(x_train, y_train, x_test, K):

   

    y_pred=[]
    for x_test_point in x_test:
      if  i == "E":
          distance_point  = distance_ecu(x_train, x_test_point)
      elif i == "M":
          distance_point = distance_M(x_train, x_test_point)
      df_nearest_point= nearest_neighbors(distance_point, K)  
      y_pred_point    = voting(df_nearest_point, y_train) 
      y_pred.append(y_pred_point)

    return y_pred 



K=3
y_pred_scratch= KNN_from_scratch(normalized_x_train, y_train, normalized_x_test, K)



knn=KNeighborsClassifier(K)
knn.fit(normalized_x_train, y_train)
y_pred_sklearn= knn.predict(normalized_x_test)
print(y_pred_sklearn)



print(np.array_equal(y_pred_sklearn, y_pred_scratch))



print(f'The accuracy of our implementation is {accuracy_score(y_test, y_pred_scratch)}')
print(f'The accuracy of sklearn implementation is {accuracy_score(y_test, y_pred_sklearn)}')



n_splits= 10 
kf= KFold(n_splits= n_splits) 

accuracy_k= [] 
k_values= list(range(1,30,2)) 

for k in k_values: 
  accuracy_fold= 0
  for normalized_x_train_fold_idx, normalized_x_valid_fold_idx in  kf.split(normalized_x_train): ## Loop over the splits
      normalized_x_train_fold= normalized_x_train[normalized_x_train_fold_idx] ## fetch the values
      y_train_fold= y_train[normalized_x_train_fold_idx]

      normalized_x_test_fold= normalized_x_train[normalized_x_valid_fold_idx]
      y_valid_fold= y_train[normalized_x_valid_fold_idx]
      y_pred_fold= KNN_from_scratch(normalized_x_train_fold, y_train_fold, normalized_x_test_fold, k)

      accuracy_fold+= accuracy_score (y_pred_fold, y_valid_fold) ## Accumulate the accuracy
  accuracy_fold= accuracy_fold/ n_splits ## Divide by the number of splits
  accuracy_k.append(accuracy_fold)

  ####
  print(f'The accuracy for each K value was {list ( zip (accuracy_k, k_values))}') 
  
  #####
  print(f'Best accuracy was {np.max(accuracy_k)}')

with open('acuracy.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(accuracy_k)
  






