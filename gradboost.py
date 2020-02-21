import numpy as np
import pandas as pd
from  sklearn.ensemble import RandomForestClassifier as RFC
from  sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import r2_score
from  sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pickle 
import matplotlib.pyplot as plt


data = pd.read_csv('gbm-data.csv')
print(data.shape)
y = np.array(data[data.keys()[0]])

X = np.array(data.drop((data.keys()[0]), axis = 1))
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.8, random_state =  241)


learn_rate = [0.2]

best_boost = 161
min_log_test, min_log_train = 15 , 15 
log_loss_test,log_loss_train , index = 0, 0, 0
for k in learn_rate:
    model = GBC(n_estimators = 250, verbose = True, random_state = 241, learning_rate = k)
    model = model.fit(X_train,y_train)
    with open('data.txt', 'wb') as file:
        pickle.dump(model, file)
##    model = None
##    with open('data.txt', 'rb') as file:
##        model = pickle.load(file)
##        
    stage_train = model.staged_decision_function(X_train)
    stage_test = model.staged_decision_function(X_test)
    ##print(len(list(stage_train)))
    st_tr_loss = []
    st_ts_loss = []
    for i, pred in enumerate(stage_train):
        
        pred = list(map(lambda y_pred: 1/(1 + np.exp(-y_pred)) , pred))
        st_tr_loss.append((log_loss(y_train,pred)))
    ##print(len(st_tr_loss))
    
    for i, pred in enumerate(stage_test):
        pred = list(map(lambda y_pred: 1/(1 + np.exp(-y_pred)) , pred)) 
        st_ts_loss.append((log_loss(y_test,pred)))
        
    for i in range(len(st_tr_loss)):
        if st_tr_loss[i] < min_log_train:
            min_log_train = st_tr_loss[i]
            log_loss_train = i
            
    for i in range(len(st_ts_loss)):
        if st_ts_loss[i] < min_log_test:
            min_log_test = st_ts_loss[i]
            log_loss_test = i
            
    print("Min Loss Train  : ",k," ",np.argmin(st_tr_loss)," ",min(st_tr_loss))
    print("Min Loss Test   : ",k," ",np.argmin(st_ts_loss)," ",min(st_ts_loss))
    if k ==0.2:
        best_boost = log_loss_test
    print(st_tr_loss)
    print(st_ts_loss)
    plt.figure()
    plt.plot(st_ts_loss, 'r', linewidth=2)
    plt.plot(st_tr_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()
print(best_boost)

forest = RFC(n_estimators = best_boost + 1, random_state = 241)
forest.fit(X_train, y_train)
y_pred = forest.predict_proba(X_test)
print("Forest Loss: ",log_loss(y_test,y_pred))
