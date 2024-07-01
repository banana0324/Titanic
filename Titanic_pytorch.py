import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import missingno as msno

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('train.csv') 
test_df = pd.read_csv('test.csv') 
submission_df = pd.read_csv('gender_submission.csv')
result_df = test_df[["PassengerId"]]

# EDA

# print(train_df.shape)
# print(test_df.shape)

# msno.matrix(train_df)
# msno.matrix(test_df)
# plt.show()
# print(train_df.head())
# print(train_df.info())

test_df[test_df.Fare.isna()]
test_df.Fare.fillna(test_df.Fare[(test_df.Age) > 55 & (test_df.Age < 65)].mean(), inplace=True)
# test_df.info()
# submission_df.head()
# sns.pairplot(data=train_df)
# plt.show()

# sns.barplot(x="Sex", y= "Survived", data=train_df)
# sns.barplot(y="Age", x="Survived", data=train_df)
# sns.barplot(x="Pclass", y= "Survived", data=train_df)
# sns.barplot(
#     x="Embarked",
#     y="Survived",
#     palette='hls',
#     data=train_df)
# sns.barplot(x="Pclass", y="Survived", hue="Embarked", data=train_df)
# sns.barplot(y="SibSp", x="Survived", data=train_df)
# sns.barplot(y="Parch", x="Survived", data=train_df)
# sns.heatmap(train_df.corr(), cmap="BrBG", vmin=-1, vmax=1, annot=True)
# heatmap = sns.heatmap(train_df.corr()[['Survived']].sort_values(by='Survived', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
# heatmap.set_title('Features Correlating with Survived', fontdict={'fontsize':18}, pad=16);


# Data integrity check
column_targets = ["Survived"]
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
all_columns = column_targets + features

def nan_checking():
    null_columns = {}
    for feature in all_columns:
        nulls = train_df[feature].isnull().sum()
        if nulls > 0:
            null_columns[feature] = nulls
    print(null_columns)

train_df.drop(labels=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True, axis=1)
test_df.drop(labels=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True, axis=1)
train_df.head()

targets_df = train_df[column_targets]
nan_checking()

def one_hot_encoding(df):
    df['Sex'].replace(['male','female'],[0,1],inplace=True)
    df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace = True)
    df['Embarked'].fillna('S',inplace = True)

fillna(train_df)
one_hot_encoding(train_df)
fillna(test_df)
one_hot_encoding(test_df)


nan_checking()
train_df.drop(labels=["Survived"], inplace=True, axis=1)

# Feature engineering

train_df2 = train_df.copy()
train_df2["Family"] = train_df2.loc[:,"SibSp":"Parch"].sum(axis=1)
train_df2.drop(labels=["SibSp","Parch"],axis=1,inplace=True)
train_df3 = train_df.copy()
train_df3.drop(labels=['SibSp'],axis=1,inplace=True)

test_df2 = test_df.copy()
test_df2["Family"] = test_df2.loc[:,"SibSp":"Parch"].sum(axis=1)
test_df2.drop(labels=["SibSp","Parch"],axis=1,inplace=True)
test_df3 = test_df.copy()
test_df3.drop(labels=['SibSp'],axis=1,inplace=True)

# print(train_df.head())
# print(train_df2.head())
# print(test_df3.head())
datadict = {
    "all":[train_df,test_df],
    "family":[train_df2,test_df2],
    "sibsp":[train_df3,test_df3]
}
# print(datadict)

# msno.matrix(train_df)
# msno.matrix(test_df)
# plt.show()
data_state = [ "all", "family", "sibsp"]
STATE = 1
train_df, test_df = datadict.get(data_state[STATE])

double_targets = [
    [j-1, j] if j == 1 else [j+1, j]
    for j in [targets_df.iloc[i, 0] for i in range(targets_df.shape[0])]
]


double_targets = np.array(double_targets)

TEST_SIZE = 0.33
RANDOM_STATE = 356 # 60
NORMALIZATION = True

if NORMALIZATION:
    normal = MinMaxScaler().fit(train_df)
    train_df = normal.transform(train_df)
    test_df = normal.transform(test_df)

else:
    scaler = StandardScaler().fit(train_df)
    train_df = scaler.transform(train_df)
    test_df = scaler.transform(test_df)

x_train, x_test, y_train, y_test = train_test_split(train_df, double_targets, shuffle=True, random_state=RANDOM_STATE, test_size=TEST_SIZE)

# Models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
x_tensor = torch.from_numpy(x_train.astype(np.float32)).to(device=device)
y_tensor = torch.from_numpy(y_train.astype(np.float32)).to(device=device)

class TitanicDataset(Dataset):
    def __init__(self,x_tensor,y_tensor):
        super().__init__()
        self.x_train = x_tensor
        self.y_train = y_tensor
        self.n_samples = x_train.shape[0]

    def __getitem__(self, index):
        return (self.x_train[index], self.y_train[index])
    
    def __len__(self):
        return len(self.n_samples)
    
dataset = TitanicDataset(x_tensor, y_tensor)
dataset  = TensorDataset(x_tensor, y_tensor)

EPOCH = 22 # 37
BATCH_SIZE = 60
ITERS = len(dataset)

dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

INPUT = x_train.shape[1]
HIDDEN_1 = 16
HIDDEN_2 = 16
HIDDEN_3 = 16
OUTPUT = 2
ALPHA = 0.01
N_EPOCHS = 22

class TitanicNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_hidden = nn.Linear(x_train.shape[1],16)
        self.hidden_hidden = nn.Linear(16,16)
        self.hidden_hidden_2 = nn.Linear(16,16)
        self.hidden_output = nn.Linear(16,2)

    def forward(self,x):
        x = F.leaky_relu(self.input_hidden(x))
        x = F.leaky_relu(self.hidden_hidden(x))
        x = F.leaky_relu(self.hidden_hidden_2(x))
        x = F.dropout(x, p = 0.28)
        y_predict = self.hidden_output(x)
        return y_predict
    
model = TitanicNeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)
    
def calc_accuracy(x_tensor, y_tensor):
    with torch.no_grad():
        model.eval()
        x_predict = model(x_tensor).argmax(axis=1)
    y_test_max = y_tensor.argmax(axis=1)
    accuracy = torch.eq(x_predict,y_test_max).to(torch.int8).sum() / len(y_test_max)
    return accuracy

accuracy_list = []
for epoch in range(N_EPOCHS):

    for iters,(x,y) in enumerate(dataloader):

        yhat = model(x)
        loss = criterion(yhat,y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        accuracy = calc_accuracy(x_tensor,y_tensor)
        accuracy_list.append(accuracy)

    if epoch % 2 == 0:
        print(f">> Epoch ~ [ {epoch+1} ] || >> Loss ~ [ {loss.item():.4f} ] || >> Accuracy ~ [ {accuracy:.2f} ]")

plt.plot([i.to(device=torch.device("cpu")) for i in accuracy_list])
# plt.show()

data = torch.from_numpy(test_df.astype(np.float32)).to(device=device)
target_column = model(data).detach().argmax(axis=1).to(dtype=torch.int64)
target_column = target_column.cpu().numpy()
result_df["Survived"] = target_column
result_df.to_csv("result_sub_2.csv", index=False)