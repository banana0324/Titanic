import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


file_path = 'Titanic.csv'

data = pd.read_csv(file_path)

titanic = pd.DataFrame(data).copy()
titanic = titanic.dropna(subset=['Embarked'])
titanic['Cabin'] = titanic['Cabin'].fillna('Not_Found')
features = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X = titanic[features]
y = titanic.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# 做one-hot encoding
features_dummy = titanic[['Embarked','Sex']]
features_dummy = pd.get_dummies(features_dummy,columns=['Embarked','Sex'])
X_train = pd.concat([X_train ,features_dummy], axis=1, join='left')


print(X_train)