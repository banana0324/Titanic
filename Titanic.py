import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

file_path = 'Titanic.csv'

data = pd.read_csv(file_path)

titanic = pd.DataFrame(data).copy()
titanic = titanic.dropna(subset=['Embarked'])
titanic['Cabin'] = titanic['Cabin'].fillna('Not_Found')

def EDA():
    print(titanic.head())
    print(titanic.shape)
    print(titanic.info())
    print(titanic.describe)
# EDA()

def dataprocessing():
    df_null = round(100*(titanic.isnull().sum())/len(titanic), 2)
    print(df_null)
# dataprocessing()

def pieplot():
    gender = titanic.groupby("Sex").agg('count')
    print(gender['PassengerId'])

    plt.pie(gender['PassengerId'], labels=['female','male'], autopct='%1.1f%%', startangle=100, colors=['pink','skyblue'])
    plt.show()
# --------------------------------------------------------------------------------------
sb.scatterplot(x = "Sex",y = "Age",data=titanic,kind = 'box')

def histogramplot():
    age = titanic.Age
    print(age)
# histogramplot()


# def barchat():