import pandas
import sklearn
from sklearn.model_selection import KFold , cross_val_score
from sklearn.preprocessing import scale 
from sklearn.metrics import roc_auc_score 
import numpy as np

def sigmoid(x, w):
    return 1./(1 + np.exp(-np.dot(x,w)))


def gradient_step( k, C, X, y, w):
    L = len(X)
    neww = []
    for i in range(len(w)):
        neww.append(w[i] + k*1/L*sum_log_i(X,w,y,i) - k*C*w[i])
    
    return neww

def sum_log_i(X,w,y,j):
    summ = 0
    for i in range(len(y)):
        summ+=y[i]*X[i][j]*(1 - 1./(1 + np.exp(-y[i]*np.dot(w,X[i]))))
    return summ

def dist(x,y):
    summ=0
    for i in range(len(x)):
        summ+=(x[i] - y[i])**2
    return np.sqrt(summ)



def Logistic_gradient(X, y, k, C):
    w = [50, 50]
    neww = [1,1]
    itercount = 0
    while dist(w , neww) >= 1e-5 :
        w = neww
        neww = gradient_step(k,C,X,y,w)
        itercount+=1
        
        if itercount > 10000:
            print("To much iteration ")
            exit(0)
    print("Itercount: ",itercount)
    return neww

X = np.array(pandas.read_csv('data-logistic.csv', header = None).iloc[:,1:])
y = np.array(pandas.read_csv('data-logistic.csv', header = None).iloc[:,0])

y+=1
y = y/2

W = Logistic_gradient(X,y, 0.2, 0)
W_reg = Logistic_gradient(X,y, 0.2, 10)


P1 = []
for i in range(len(X)):
    P1.append(sigmoid(X[i],W))

P2 = []
for i in range(len(X)):
    P2.append(sigmoid(X[i],W_reg))
print(len(P1))
score1 = roc_auc_score(y,P1)
score2 = roc_auc_score(y,P2)

print("Answer: ",score1,score2)






        
