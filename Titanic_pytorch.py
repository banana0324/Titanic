import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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