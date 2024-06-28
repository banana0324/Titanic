import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')
df_sub   = pd.read_csv('gender_submission.csv')

df_train.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
df_test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

sex = pd.get_dummies(df_train['Sex'],drop_first=True)
embark = pd.get_dummies(df_train['Embarked'],drop_first=True)
df_train = pd.concat([df_train,sex,embark],axis=1)

df_train.drop(['Sex','Embarked'],axis=1,inplace=True)

sex = pd.get_dummies(df_test['Sex'],drop_first=True)
embark = pd.get_dummies(df_test['Embarked'],drop_first=True)
df_test = pd.concat([df_test,sex,embark],axis=1)

df_test.drop(['Sex','Embarked'],axis=1,inplace=True)

Scaler1 = StandardScaler()
Scaler2 = StandardScaler()

train_columns = df_train.columns
test_columns = df_test.columns

df_train = pd.DataFrame(Scaler1.fit_transform(df_train))
df_test = pd.DataFrame(Scaler2.fit_transform(df_test))

df_train.columns = train_columns
df_test.columns = test_columns

features = df_train.iloc[:,:2].columns.to_list()
target = df_train['Survived'].name

x_train = df_train.iloc[:,:2].values
y_train = df_train['Survived'].values

x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

model = Net()
print(model)

lr = 1e-1
n_epochs = 500

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

batch_size = 64

batch_no = len(x_train) // batch_size

train_loss = 0
train_loss_min = np.Inf

def make_train_step(model, loss_fn, optimizer):

    def train_step(x, y):

        model.train()
        yhat = model(x)
        loss = loss_fn(y,yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


for epoch in range(n_epochs):
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        # x_var = Variable(torch.FloatTensor(x_train[start:end]))
        # y_var = Variable(torch.LongTensor(y_train[start:end]))
        x_var = x_train_tensor[start:end]
        y_var = y_train_tensor[start:end]

        optimizer.zero_grad()
        output = model(x_var)
        loss = loss_fn(output,y_var)
        loss.backward()
        optimizer.step()

        values, label = torch.max(output, 1)
        num_right = np.sum(label.data.numpy() == y_train[start:end])
        train_loss += loss.item() * batch_size

    train_loss = train_loss / len(x_train)
    if train_loss <= train_loss_min:
        print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(train_loss_min,train_loss))
        torch.save(model.state_dict(), "model.pt")
        train_loss_min = train_loss

    if epoch % 200 == 0:
        print('')
        print("Epoch: {} \tTrain Loss: {} \tTrain Accuracy: {}".format(epoch+1, train_loss,num_right / len(y_train[start:end]) ))
print('Training Ended! ')