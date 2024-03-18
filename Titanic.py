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
# 看Age與Sex的關係
# sb.catplot(x = "Sex",y = "Age",data=titanic,kind = 'box')

# 觀察缺失資料在船艙等級(Pclass)的分布
# sb.countplot(x = titanic["Pclass"], hue = titanic['Age'].isnull())
# plt.show()

# 觀察Age和Survived的相關性
# index_survived = (titanic["Age"].isnull()==False)&(titanic["Survived"]==1)
# index_died = (titanic["Age"].isnull()==False)&(titanic["Survived"]==0)

# sb.distplot( titanic.loc[index_survived ,'Age'], bins=20, color='blue', label='Survived' )
# sb.distplot( titanic.loc[index_died ,'Age'], bins=20, color='red', label='Survived' )
# plt.show()

# Age和Name的相關性
titanic['Title'] = titanic.Name.str.split(', ', expand=True)[1]
titanic['Title'] = titanic.Title.str.split('.', expand=True)[0]
print(titanic['Title'].unique())

# 計算每個 Title 的年齡平均值
Age_Mean = titanic[['Title','Age']].groupby( by=['Title'] ).mean()

Age_Mean.columns = ['Age_Mean']
Age_Mean.reset_index( inplace=True )

titanic=titanic.reset_index() #重整index
for i in range(len(titanic["Age"].isnull())):
    if titanic["Age"].isnull()[i]==True:
        for j in range(len(Age_Mean.Title)):
            if titanic["Title"][i]==Age_Mean.Title[j]:
                titanic["Age"][i]=Age_Mean.Age_Mean[j]

dataprocessing()
def histogramplot():
    age = titanic.Age
    print(age)
# histogramplot()


# def barchat():