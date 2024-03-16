import pandas as pd
import matplotlib.pyplot as plt

file_path = 'Titanic.csv'

titanic = pd.read_csv(file_path)

data = pd.DataFrame(titanic)

def EDA():
    print(data.head())
    print(data.shape)
    print(data.info())
    print(data.describe)
# EDA()

def dataprocessing():
    data = data.dropna(subset=['Embarked'])
    df_null = round(100*(data.isnull().sum())/len(data), 2)
    print(df_null)
dataprocessing()

def pieplot():
    gender = data.groupby("Sex").agg('count')
    print(gender['PassengerId'])

    plt.pie(gender['PassengerId'], labels=['female','male'], autopct='%1.1f%%', startangle=100, colors=['pink','skyblue'])
    plt.show()

def histogramplot():
    age = data.Age
    print(age)

# histogramplot()


# def barchat():