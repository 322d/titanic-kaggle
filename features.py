import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# extracting titles from names
train_test_data = [train, test]  # combining train and test datasets

for dataset in train_test_data:
    # regex retrives alphabet values that end with a dot
    dataset['Title'] = dataset['Name'].str.extract(
        ' ([A-Za-z]+)\.', expand=False)
    # creating new feature "Title" and setting it to the title extracted
    # from the "Name" feature.

print(train['Title'].value_counts())

# we can have 4 categories: Mr, Miss, Mrs, Others
title_mapping = {"Mr": 0, "Mrs": 1, "Miss": 2}
for dataset in train_test_data:
    for title in dataset['Title']:
        if title not in ["Mr", "Mrs", "Miss"]:
            title_mapping[title] = 3

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

# removing the "Name" feature because we don't need it anymore
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

# sex mapping, male = 0, female = 1, changing titles to numeric values
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

# age is missing fields
# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train["Age"].fillna(train.groupby("Title")[
                    "Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")[
                   "Age"].transform("median"), inplace=True)

# binning age
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[dataset['Age'] > 62, 'Age'] = 4


# fill out missing embark with S embark
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# changing embark to numeric values
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

# filling missing Fare values with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")[
                     "Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")[
                    "Fare"].transform("median"), inplace=True)

# binning fare
for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[dataset['Fare'] > 100, 'Fare'] = 3

# cleaning "Cabin" feature, keeping only the letter
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

# changing Cabin to numeric values between range 0 and 3
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8,
                 "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

# fill missing Cabin with median cabin for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")[
                      "Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")[
                     "Cabin"].transform("median"), inplace=True)


# creating new feature family size
# shows that single passagers are more likely to die than families
# note: +1 for the person itself
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

# changing FamilySize to numeric values
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6,
                  6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

# dropping values which we don't need
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)

train = train.drop('PassengerId', axis=1)
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']

