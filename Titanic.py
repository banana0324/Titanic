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
# ------------------------2--------------------------------------------------------------
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
# print(titanic['Title'].apply(select_name))

# # 計算每個 Title 的年齡平均值
Age_Mean = titanic[['Title','Age']].groupby( by=['Title'] ).mean()

Age_Mean.columns = ['Age_Mean']
Age_Mean.reset_index( inplace=True )

titanic=titanic.reset_index() #重整index
for i in range(len(titanic["Age"].isnull())):
    if titanic["Age"].isnull()[i]==True:
        for j in range(len(Age_Mean.Title)):
            if titanic["Title"][i]==Age_Mean.Title[j]:
                titanic["Age"][i]=Age_Mean.Age_Mean[j]

# # dataprocessing()

# def histogramplot():
#     sb.histplot(x='Age',data=titanic)
#     plt.show()
# # histogramplot()

# def survivePlot():
#     cols = ['Sex','Pclass','SibSp','Parch','Embarked']

#     n_rows = 2
#     n_cols = 3

#     # The subplot grid and figure size of each graph
#     fig, axs = plt.subplots(n_rows,n_cols,figsize = (n_cols * 3.2, n_rows * 3.2))

#     for r in range(0, n_rows):
#         for c in range(0, n_cols):
    
#             i = r * n_cols + c  # index to go through the number of columns
#             if i < 5:
#                 ax = axs[r][c] #show where to position each sub plots
#                 sb.countplot(x = titanic[cols[i]], hue = titanic['Survived'], ax= ax)
#                 ax.set_title(cols[i])
#                 ax.legend(title = 'survived', loc = 'upper right')

#     plt.tight_layout()
#     plt.show()

# # 描述性統計
# # survivePlot()
# # 大致可以觀察出

# # 女性存活的機會比男性來得高
# # 頭等艙存活機會較高
# # 可能有帶兄弟姊妹、老婆丈夫的乘客存活機會較高
# # 有帶小孩父母親的存活機會較高
# # 從S碼頭出發有可能艙位比較低，存活機會較低

# def genderSurvive():
#     titanic.groupby('Sex')[['Survived']].mean()
#     titanic.pivot_table('Survived',index = 'Sex', columns = 'Pclass')

#     age = pd.cut(titanic['Age'],[0, 18, 80])
#     titanic.pivot_table('Survived',['Sex',age], 'Pclass')
# # genderSurvive()

# def pClassSurvive():
#     plt.scatter(titanic['Fare'],titanic['Pclass'], color = 'purple', label = 'Passenger Paid')
#     plt.ylabel('Class')
#     plt.xlabel('Price / Fare')
#     plt.title('Price of Each Class')
#     plt.legend()
#     plt.show()
# # pClassSurvive(

# # --------------------------資料分割------------------------------------
features = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X = titanic[features]
y = titanic.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# 做one-hot encoding
features_dummy = titanic[['Embarked','Sex','Title']]
features_dummy = pd.get_dummies(features_dummy,columns=['Embarked','Sex','Title'])
X_train = pd.concat([X_train,features_dummy],axis=1)
X_test = pd.concat([X_test,features_dummy],axis=1)
print(X_train)

traindf = X_train.drop(['Title_Capt', 'Title_Col', 'Title_Don', 'Title_Dr', 'Title_Jonkheer',
       'Title_Lady', 'Title_Major','Title_Mlle',
       'Title_Mme',  'Title_Ms', 'Title_Rev',
       'Title_Sir', 'Title_the Countess','Embarked','Sex'], axis=1)
testdf = X_train.drop(['Title_Capt', 'Title_Col', 'Title_Don', 'Title_Dr', 'Title_Jonkheer',
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
# trainx = traindf.drop(['Survived'], axis=1)
# 將答案當作目標資料
# trainy = traindf['Survived']

# n_estimators -> 你要有幾棵樹
# clf = RandomForestClassifier(max_depth=8, n_estimators=50)

# cv參數決定切幾份(這裡切10份) 
# 用np.average取其平均
# np.average(cross_val_score(clf, trainx, trainy, cv=10))


# #Scale the data 
# # from sklearn.preprocessing import StandardScaler
# # sc = StandardScaler()
# # X_trian = sc.fit_transform(X_train)
# # X_test = sc.transform(X_test)