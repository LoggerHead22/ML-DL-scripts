import pandas
import sklearn
from sklearn.model_selection import KFold , cross_val_score
from sklearn.preprocessing import scale 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def best_cross_val(X,y,CrVal):
    max_coef = 0
    best_k = 0
    for k in range(1,50):
        model = KNeighborsClassifier(n_neighbors = k)
        coef = cross_val_score(model,X,y,cv = CrVal)
        coef=coef.mean()
        ##print(coef)
        if(coef > max_coef):
            max_coef = coef
            best_k = k



    print(max_coef)
    print(best_k)


data = pandas.read_csv('wine.data', header = None)
print(data)

y = np.array(data[0])
print(y)
data = data.drop([0], axis = 1)
X = np.array(data)
print(len(X))

CrVal = sklearn.model_selection.KFold(n_splits = 5, shuffle = True, random_state = 42)

best_cross_val(X,y,CrVal)

print(X)
X = scale(X)
print(X)

best_cross_val(X,y,CrVal)
