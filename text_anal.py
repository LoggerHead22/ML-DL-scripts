import sklearn.datasets
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
data= sklearn.datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism','sci.space'])


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data.data)
y = data.target
print(X[1])
grid = {'C': np.power(10.0 , np.arange(-5,6))}
cv = KFold(random_state = 241, shuffle = True, n_splits = 5 )
model = SVC(kernel = 'linear' , random_state = 241)
##gs = GridSearchCV(model,grid,scoring='accuracy',cv = cv)

print("do")
##gs.fit(X,y)
print("posle")

##best_C = gs.best_params_['C']

best_C = 1
print(best_C)
model = SVC(kernel = 'linear' , C = best_C, random_state = 241)
model.fit(X,y)

word = vectorizer.get_feature_names()

feature = model.coef_
print(feature)
feature = np.absolute(feature.getrow(0).toarray()[0].ravel())
print(feature)
feature = np.argsort(feature)[-10:]
array = [] 
for i in range(10):
    array.append(word[feature[i]])
    print(array[i])
array.sort()

print(array)



    
