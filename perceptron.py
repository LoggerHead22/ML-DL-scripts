import pandas
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score 
data_train = pandas.read_csv('perceptron-train.csv', header = None )
data_test = pandas.read_csv('perceptron-test.csv', header = None )

y_train = np.array(data_train[0])
y_test = np.array(data_test[0])

X_train = data_train.drop([0], axis = 1)
X_test = data_test.drop([0], axis = 1)


model = Perceptron(random_state=241,max_iter=5, tol=None)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

score1 = accuracy_score(y_test, y_predict)

print("Score 1: ",score1)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

model.fit(X_train_scaler,y_train)
y_pred_scal = model.predict(X_test_scaler)

score2 = accuracy_score(y_test,y_pred_scal)

print("Score 2: ", score2)
print("Answer: ",score2 - score1)
