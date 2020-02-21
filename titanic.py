import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from collections import Counter
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

def Quation():
    print(data['Sex'].value_counts())
    count = data.count()
    print(data.count())

    print(data['Age'].median())

    print(data['SibSp'].corr(data['Parch']))

    print(data.corr())

    name = data.sort_values(by = ['Sex'])['Name'].tolist()

    name = name[0:314]
    print(len(name))
    for i in range(len(name)):
        a = name[i].find('Miss. ')
        ##print(a)
        if(a!=-1):
          ##  print(name[i])
            name[i] = name[i][a:].split()[1]
          ##  print(name[i])
        else:
            b = name[i].find('(')
          ##  print(name[i])
            if(b!=-1):
                name[i] = name[i][b + 1:].split()[0]
                name[i] = name[i].replace(')',"")
            else:
                name[i] =""
          ##  print(name[i])

    Max = Counter(name)
    print(Max.most_common(1)[0])

    
data['Sex'] = list(map(lambda x: 0 if x == 'male' else 1, data['Sex']))

##print(data['Sex'])


data = data[['Pclass','Age','Sex','Fare','Survived']]
data = data.dropna(axis = 0)
##print(data['Age'])
X = np.array(data[['Pclass','Age','Sex','Fare']])

y = np.array(data['Survived'])
print(len(y) , len(X))

tree = DecisionTreeClassifier( random_state = 241)

tree.fit(X,y)
importances = tree.feature_importances_
print(importances)
##print(X)


##print(Max)



