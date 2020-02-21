import sklearn.datasets
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold , cross_val_score
from sklearn.neighbors import KNeighborsRegressor
data = sklearn.datasets.load_boston()

def best_cross_val_metric(X,y,CrVal):
    max_coef = -100
    best_p = 0
    pp=1
    a = np.linspace(1,50,num=200)
    for pp in a:
        model = KNeighborsRegressor(n_neighbors = 5, weights='distance', metric = 'minkowski',p=pp)
        coef = cross_val_score(model,X,y,cv = CrVal,scoring = 'neg_mean_squared_error')
        coef=coef.mean()
        print(coef)
        if(coef > max_coef):
            max_coef = coef
            best_p = pp
        pp+=0.25
    print(best_p)
    
X = data.data
X = scale(X)
y = data.target

print(X)
print(y)
CrVal = KFold(random_state = 42, shuffle = True, n_splits = 5 )



best_cross_val_metric(X,y,CrVal)
