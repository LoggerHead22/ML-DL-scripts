import pandas
import numpy as np
import sklearn.metrics as sm
from collections import Counter


def prec_70(curve):
    maxx = 0
    for i in range(len(curve[0])):
        if curve[1][i] >=0.7 and curve[0][i] > maxx:
            maxx=curve[0][i]
    return maxx


data = pandas.read_csv('classification.csv')

print(data['true'].value_counts())
print(data['pred'].value_counts())

true = np.array(data['true'])
pred = np.array(data['pred'])

true = true * 2

matr = Counter(true - pred)
print(matr)
print("TP: ",matr[1], "FP: ",matr[-1],"FN: ",matr[2],"TN: ",matr[0])

true = true/2

accur = sm.accuracy_score(true,pred)
precision = sm.precision_score(true, pred)
recall = sm.recall_score(true,pred)
F = sm.f1_score(true,pred)

print(accur, precision, recall, F)

data = pandas.read_csv('scores.csv')
true = np.array(data['true'])
algh = {'log' : np.array(data['score_logreg']) , 'svm' : np.array(data['score_svm']), 'knn' : np.array(data['score_knn']),'tree' : np.array(data['score_tree'])}

auc_roc = []
auc_roc.append(sm.roc_auc_score(true,algh['log']))
auc_roc.append(sm.roc_auc_score(true,algh['svm']))
auc_roc.append(sm.roc_auc_score(true,algh['knn']))
auc_roc.append(sm.roc_auc_score(true,algh['tree']))
print("AUC_ROC: ",auc_roc)

curve_log = sm.precision_recall_curve(true,algh['log'])
curve_svm = sm.precision_recall_curve(true,algh['svm'])
curve_knn = sm.precision_recall_curve(true,algh['knn'])
curve_tree = sm.precision_recall_curve(true,algh['tree'])


print("log ",prec_70(curve_log))
print("svm ",prec_70(curve_svm))
print("knn ",prec_70(curve_knn))
print("tree ",prec_70(curve_tree))





