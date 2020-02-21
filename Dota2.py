import pandas 
import numpy as np
from  sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.model_selection import KFold , cross_val_score
import time 
import datetime
from threading import Thread
from  sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale 
import warnings
from sklearn.metrics import roc_auc_score 
warnings.filterwarnings("ignore")



def GradboostCV(k):
  start_time = datetime.datetime.now()
  model = GBC(n_estimators = k,  random_state = 241)
  coef = cross_val_score(model, X, y, cv = CrVal,scoring = 'roc_auc')
  coef=coef.mean()
  grboost_coef.append([k,round(coef,2), str(datetime.datetime.now() - start_time)] )

def LogRegThread_C(C_in, X_in):
  start_time = datetime.datetime.now()
  model = LogisticRegression(penalty = 'l2',C = C_in ,random_state = 241)
  coef = cross_val_score(model, X_in, y, cv = CrVal,scoring = 'roc_auc')
  coef=coef.mean()
  logreg_coef.append([C_in,round(coef,2), str(datetime.datetime.now() - start_time)] )

def LogRegThread(X_in):
  full_time = datetime.datetime.now()
  
  thread1 = Thread(target =LogRegThread_C, args = (0.01,X_in), )
  thread2 = Thread(target =LogRegThread_C, args = (0.1,X_in), )
  thread3 = Thread(target =LogRegThread_C, args = (10,X_in), )
  thread4 = Thread(target =LogRegThread_C, args = (100,X_in), )
  
  thread1.start()
  thread2.start()
  thread3.start()
  thread4.start()
  
  thread1.join()
  thread2.join()
  thread3.join()
  thread4.join()

  print("\n\n LogisticRegressor\n",*logreg_coef,sep = "\n")
  print("FULL TIME: ",datetime.datetime.now() - full_time)

  

#%%
features = pandas.read_csv('features/features.csv', index_col = 'match_id')
test = pandas.read_csv('features_test/features_test.csv', index_col = 'match_id')
print(features.head())

y = np.array(features['radiant_win'])
X = features.drop(features.columns[102:108] , axis = 1)
X_test = test.drop(test.columns[102:108] , axis = 1)
X_count = np.array(X.count())
X_names = np.array(X.keys().tolist())
X_nan = list(zip(X_names[np.where(X_count!= X.shape[0])],X.shape[0] - X_count[np.where(X_count!= X.shape[0])]))

print(X_nan)


#%%
X.fillna(value = 0 , inplace = True )
X_test.fillna(value = 0 , inplace = True )
CrVal = KFold(random_state = 42, shuffle = True, n_splits = 5 )

grboost_coef = [] 
full_time = datetime.datetime.now()

thread1 = Thread(target =GradboostCV, args = (10,) )
thread2 = Thread(target =GradboostCV, args = (20,) )
thread3 = Thread(target =GradboostCV, args = (30,) )
##thread4 = Thread(target =GradboostCV, args = (40,) )

thread1.start()
thread2.start()
thread3.start()
##thread4.start()

thread1.join()
thread2.join()
thread3.join()
##thread4.join()

print("\n\n GradientBoosting\n",*grboost_coef,sep = "\n")
print("FULL TIME: ",datetime.datetime.now() - full_time)

#%%
X_scale = scale(X)

logreg_coef = []

LogRegThread(X_scale)
#%%

Categ_ind =  np.array(list(map( lambda name : 1 if name=='lobby_type' or name.endswith('hero') else 0 , X_names))) 
Categ_ind = np.where(Categ_ind == 1)

X_nocategory = X.drop(X.columns[Categ_ind], axis = 1)
X_nocategory_test = X_test.drop(X_test.columns[Categ_ind], axis = 1)

X_hero = X[X.columns[Categ_ind]]
X_hero.drop(X_hero.columns[0], axis = 1, inplace = True)
X_hero_test = X_test[X_test.columns[Categ_ind]]
X_hero_test.drop(X_hero_test.columns[0], axis = 1, inplace = True)


logreg_coef = []
LogRegThread(scale(X_nocategory))

#%%
hero_count = np.unique(X_hero.values)
print(len(hero_count))
X_pick = np.zeros((X_hero.shape[0], len(hero_count) + 4))
X_pick_test = np.zeros((X_hero_test.shape[0], len(hero_count) + 4))

for i, match_id in enumerate(X_hero.index):
    for p in range(5):
        X_pick[i, X_hero.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, X_hero.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
       
for i, match_id in enumerate(X_hero_test.index):
    for p in range(5):
        X_pick_test[i, X_hero_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick_test[i, X_hero_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
        
        
        
        
X_pick = X_pick[:,hero_count - 1] 
X_pick_test = X_pick_test[:,hero_count - 1] 
#%%
X_final = np.hstack((X_pick, np.array(X_nocategory)))
X_final_test = np.hstack((X_pick_test, np.array(X_nocategory_test)))
print(X_final.shape)

logreg_coef = []
LogRegThread(scale(X_final))

model = LogisticRegression(penalty = 'l2',C = 1 ,random_state = 241)
model.fit(scale(X_final),y)
y_pred = model.predict_proba(scale(X_final_test))[:,1]

      

