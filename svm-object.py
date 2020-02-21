import pandas
import sklearn
from sklearn.model_selection import KFold , cross_val_score
from sklearn.preprocessing import scale 
from sklearn.svm import SVC
import numpy as np


X = pandas.read_csv('svm-data.csv', header = None).iloc[:,1:]
y = pandas.read_csv('svm-data.csv', header = None).iloc[:,0]

model = SVC(random_state = 241, kernel='linear', C = 100000)
model.fit(X,y)

print(model.support_)
