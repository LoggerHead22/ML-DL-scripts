import sklearn.datasets
import numpy as np
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import  hstack, coo_matrix

 
data = pandas.read_csv('salary-train.csv')
data_test = pandas.read_csv('salary-test-mini.csv')
print(data.count())


data['FullDescription'] = data['FullDescription'].str.lower()##нижний регистр 
data_test['FullDescription'] = data_test['FullDescription'].str.lower()

data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)##заменить все кроме букв и цифр
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
vectorizer = TfidfVectorizer(min_df = 5)

X_1 = data['FullDescription'] = vectorizer.fit_transform(data['FullDescription'])
X_1_test = data_test['FullDescription'] = vectorizer.transform(data_test['FullDescription'])


data['LocationNormalized'].fillna('nan', inplace=True)##заменяем прпуски 
data['ContractTime'].fillna('nan', inplace=True)
data_test['LocationNormalized'].fillna('nan', inplace=True)##заменяем прпуски 
data_test['ContractTime'].fillna('nan', inplace=True)



enc = DictVectorizer()
X_2_3= enc.fit_transform(data[['LocationNormalized','ContractTime']].to_dict('records'))##закодировали столбцы 
X_2_3_test = enc.transform(data_test[['LocationNormalized','ContractTime']].to_dict('records'))


X_train = hstack([X_1, X_2_3]) ## объеденяем разряженные матрицы 
y_train = np.array(data['SalaryNormalized'])

X_test = hstack([X_1_test, X_2_3_test])

model = Ridge(alpha = 1, random_state = 41)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)





