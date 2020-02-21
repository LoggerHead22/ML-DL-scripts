import numpy as np
import pandas as pd
from  sklearn.ensemble import RandomForestRegressor as RFS
from sklearn.metrics import r2_score
from  sklearn.model_selection import KFold
data = pd.read_csv('abalone.csv')

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = np.array(data.drop(['Rings'], axis = 1))
y = np.array(data['Rings'])

print(y)
kf = KFold(n_splits = 5, shuffle = True , random_state = 1)

train_ind = []
test_ind = []
for train_index, test_index in kf.split(X):
    train_ind.append(train_index)
    test_ind.append(test_index)

print(len(train_index))

maxscore , flag = 0 , 0
score = 0
best_k = 0
for k in range(1,50):
    model = RFS( n_estimators = k, random_state = 1)
    for i in range(1,5):   
        model.fit(X[train_ind[i]],y[train_ind[i]])
        y_pred = model.predict(X[test_ind[i]])
        score +=r2_score(y[test_ind[i]], y_pred)
    score/=5
    if round(score,2) > 0.52 and flag==0 :
        print("First k :",k)
        flag = 1
    if flag!=0:
        print("K: ",k," score : ",score)



