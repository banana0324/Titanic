import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


file_path = 'Titanic.csv'

data = pd.read_csv(file_path)
data = data.reset_index(drop=True)
titanic = pd.DataFrame(data).copy()
titanic = titanic.dropna(subset=['Embarked'])
titanic['Cabin'] = titanic['Cabin'].fillna('Not_Found')

def EDA():
    print(titanic.head())
    print(titanic.shape)
    print(titanic.info())
    print(titanic.describe())
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
# pieplot()
# ------------------------2--------------------------------------------------------------
# 看Age與Sex的關係
# sb.catplot(x = "Sex",y = "Age",data=titanic,kind = 'box')
# plt.show()
# 觀察缺失資料在船艙等級(Pclass)的分布
# sb.countplot(x = titanic["Pclass"], hue = titanic['Age'].isnull())
# plt.show()

# 觀察Age和Survived的相關性
# index_survived = (titanic["Age"].isnull()==False)&(titanic["Survived"]==1)
# index_died = (titanic["Age"].isnull()==False)&(titanic["Survived"]==0)
# sb.distplot( titanic.loc[index_survived ,'Age'], bins=20, color='blue', label='Survived' )
# sb.distplot( titanic.loc[index_died ,'Age'], bins=20, color='red', label='Survived' )
# plt.show()

# Age和Name的相關性(稱謂處理)
titanic['Title'] = titanic.Name.str.split(', ', expand=True)[1]
titanic['Title'] = titanic.Title.str.split('.', expand=True)[0]
# print(titanic['Title'].value_counts())
# print(titanic['Title'].unique())
def select_name(s):
    d = {
        'Master' : 'Master',
        'Miss' : 'Miss',
        'Mr' : 'Mr',
        'Mrs' : 'Mrs'
    }
    return d.get(s)
name = titanic['Title'].apply(select_name)
dummy = pd.get_dummies(name)
traindf = pd.concat([titanic, dummy], axis=1)
print(traindf)

# 計算每個 Title 的年齡平均值
Age_Mean = titanic[['Title','Age']].groupby( by=['Title'] ).mean()
Age_Mean.columns = ['Age_Mean']
Age_Mean.reset_index( inplace=True )
# print(Age_Mean)

# Title 年齡補值
titanic=titanic.reset_index()
for i in range(len(titanic["Age"].isnull())):
    if titanic["Age"].isnull()[i]==True:
        for j in range(len(Age_Mean.Title)):
            if titanic["Title"][i]==Age_Mean.Title[j]:
                titanic["Age"][i]=Age_Mean.Age_Mean[j]

# dataprocessing()

def survivePlot():
    cols = ['Sex','Pclass','SibSp','Parch','Embarked']

    n_rows = 2
    n_cols = 3

    # The subplot grid and figure size of each graph
    fig, axs = plt.subplots(n_rows,n_cols,figsize = (n_cols * 3.2, n_rows * 3.2))

    for r in range(0, n_rows):
        for c in range(0, n_cols):
    
            i = r * n_cols + c  # index to go through the number of columns
            if i < 5:
                ax = axs[r][c] #show where to position each sub plots
                sb.countplot(x = titanic[cols[i]], hue = titanic['Survived'], ax= ax)
                ax.set_title(cols[i])
                ax.legend(title = 'survived', loc = 'upper right')

    plt.tight_layout()
    plt.show()

# survivePlot()


def genderSurvive():
    titanic.groupby('Sex')[['Survived']].mean()
    titanic.pivot_table('Survived',index = 'Sex', columns = 'Pclass')

    age = pd.cut(titanic['Age'],[0, 18, 40, 80])
    print(titanic.pivot_table('Survived',['Sex',age], 'Pclass'))
# genderSurvive()

def pClassSurvive():
    plt.scatter(titanic['Fare'],titanic['Pclass'], color = 'blue', label = 'Passenger Paid')
    plt.ylabel('Class')
    plt.xlabel('Price / Fare')
    plt.title('Price of Each Class')
    plt.legend()
    plt.show()
# pClassSurvive()

# # --------------------------資料分割------------------------------------
features = ['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']
X = titanic[features]
y = titanic.Survived

# 做one-hot encoding
features_dummy = titanic[['Embarked','Sex','Title']]
features_dummy = pd.get_dummies(features_dummy,columns=['Embarked','Sex','Title'])
X = pd.concat([X ,features_dummy],axis=1)
# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


traindf = X_train.drop(['PassengerId','Title','Title_Capt', 'Title_Col', 'Title_Don', 'Title_Dr', 'Title_Jonkheer',
       'Title_Lady', 'Title_Major','Title_Mlle',
       'Title_Mme',  'Title_Ms', 'Title_Rev',
       'Title_Sir', 'Title_the Countess','Embarked','Sex'], axis=1)
testdf = X_test.drop(['Title','Survived','Title_Capt', 'Title_Col', 'Title_Don', 'Title_Dr', 'Title_Jonkheer',
       'Title_Lady', 'Title_Major','Title_Mlle',
       'Title_Mme',  'Title_Ms', 'Title_Rev', 
       'Title_Sir', 'Title_the Countess','Embarked','Sex'], axis=1)


# 相關係數
# print(traindf.corr())
plt.figure(figsize=(14, 14))

sb.heatmap(traindf.corr(), annot=True, cmap='RdBu')
# plt.show()

# --------------------------建模--------------------------------
# 交叉驗證(Cross Validation)
# 假設9/1切train_test_split
# 每次拿其中1份驗證，共驗證10次。
from sklearn.model_selection import cross_val_score
# numpy > 處理大量數字
import numpy as np

# 把訓練資料的答案丟掉
trainx = traindf.drop(['Survived'], axis=1)
# 將答案當作目標資料
trainy = traindf['Survived']

# n_estimators -> 你要有幾棵樹
model = RandomForestClassifier(max_depth=8, n_estimators=50)

# cv參數決定切幾份(這裡切10份) 
# 用np.average取其平均
print(np.average(cross_val_score(model, trainx, trainy, cv=10)))


# 貪婪搜索參數(一個一個試)
# 找max_depth與n_estimators
from sklearn.model_selection import GridSearchCV
p = {
#   深度的搜索範圍(5~10)
    'max_depth':list(range(5, 11)),
#   棵數的搜索範圍(20~30)
    'n_estimators':list(range(20, 31))
}
model = RandomForestClassifier()
s = GridSearchCV(model, p, cv=5)
s.fit(trainx, trainy)
# best_params_L: 最好的參數
# best_score_: 最好的分數
print(s.best_params_)
print(s.best_score_)


# 把旅客ID取出
testx = testdf.drop(['PassengerId'], axis=1)
# 存下旅客ID
testid = testdf['PassengerId']
# 設定隨機森林參數
clf = RandomForestClassifier(max_depth=7, n_estimators=24)
# 將訓練資料放入隨機森林運算
clf.fit(trainx, trainy)
# 利用訓練好的模型預測測試資料
pre = clf.predict(testx)
# 將計算結果存下，並輸出成csv檔
result = pd.DataFrame()
result['PassengerId'] = testid
result['Survived'] = pre
result.to_csv('result.csv', encoding='utf-8', index=False)

