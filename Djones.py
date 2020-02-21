import numpy as np
import pandas
from sklearn.decomposition import PCA

data = pandas.read_csv('close_prices.csv')

X = data.drop(['date'], axis = 1)

pca = PCA(n_components = 10)

pca.fit(X)

print(pca.explained_variance_ratio_)
print(pca.components_[0])

names = data.keys()
print(names)
num = pca.components_[0].argmax()
print("Name : ",names[num + 1])


X_trans = np.array(pca.transform(X))
first_comp = pandas.DataFrame(X_trans[:,0])


print(X_trans[0].shape)
djones = pandas.read_csv('djia_index.csv')

corr = np.corrcoef(first_comp.T, djones['^DJI'])
print("Corr :",corr)

