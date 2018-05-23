import os
import pandas as pd
import numpy  as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

train_df.fillna(train_df.mean(), inplace = True)
test_df.fillna(test_df.mean(),   inplace = True)

# check our ages have no nan
# print(train_df["Age"].values)

# sex feature
def get_sex(sex):
	if sex == "male":
		return 1
	else:
		return 0

train_df["Sex"] = train_df["Sex"].map(get_sex)
test_df["Sex"] = test_df["Sex"].map(get_sex)

# deck
def get_deck(deck):
	if isinstance(deck, float):
		return "N"
	elif "A" in deck:
		return "A"
	elif "B" in deck:
		return "B"
	elif "C" in deck:
		return "C"
	elif "D" in deck:
		return "D"
	elif "E" in deck:
		return "E"
	elif "F" in deck:
		return "F"
	elif "T" in deck:
		return "T"
	elif "G" in deck:
		return "G"

train_df["Deck"] = train_df["Cabin"].map(get_deck)
train_df = pd.get_dummies(train_df, columns = ["Deck"])

test_df["Deck"] = test_df["Cabin"].map(get_deck)
test_df = pd.get_dummies(test_df, columns = ["Deck"])

# has cabin
# print(train_df["Cabin"].values)
def get_has_cabin(cabin):
	if isinstance(cabin, float):
		return 0
	else:
		return 1

train_df["HasCabin"] = train_df["Cabin"].map(get_has_cabin)
test_df["HasCabin"] = test_df["Cabin"].map(get_has_cabin)

# child
def get_child(age):
	if age <= 16:
		return 1
	else:
		return 0

train_df["IsChild"] = train_df["Age"].map(get_child)
test_df["IsChild"] = test_df["Age"].map(get_child)

# family size
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

# get dummy variable for family size
def get_family_group(fam):
	if fam <= 1:
		return "Lone"
	elif fam <= 4:
		return "Small"
	else:
		return "Large"
train_df["FamilyGroup"] = train_df["FamilySize"].map(get_family_group)
train_df = pd.get_dummies(train_df, columns = ["FamilyGroup"])

test_df["FamilyGroup"] = test_df["FamilySize"].map(get_family_group)
test_df = pd.get_dummies(test_df, columns = ["FamilyGroup"])

# alone
def get_alone(family):
	if family > 1:
		return 0
	else:
		return 1

train_df["IsAlone"] = train_df["FamilySize"].map(get_alone)
test_df["IsAlone"] = test_df["FamilySize"].map(get_alone)

# fare per person
train_df["FarePerPerson"] = train_df["Fare"] / train_df["FamilySize"]
test_df["FarePerPerson"] = test_df["Fare"] / test_df["FamilySize"]

# get titles
def get_titles(name):
	if "Mrs" in name:
		return "Mrs"
	elif "Mr" in name:
		return "Mr"
	elif "Miss" in name:
		return "Miss"
	else:
		return "Rare"

train_df["Title"] = train_df["Name"].map(get_titles)
train_df = pd.get_dummies(train_df, columns = ["Title"])

test_df["Title"] = test_df["Name"].map(get_titles)
test_df = pd.get_dummies(test_df, columns = ["Title"])

# get pclass
def get_class(pclass):
	if pclass == 1:
		return "High"
	elif pclass == 2:
		return "Middle"
	else:
		return "Low"

train_df["Class"] = train_df["Pclass"].map(get_class)
train_df = pd.get_dummies(train_df, columns = ["Class"])

test_df["Class"] = test_df["Pclass"].map(get_class)
test_df = pd.get_dummies(test_df, columns = ["Class"])

# name length
def get_name_length(name):
	if len(name) <= 20:
		return "Short"
	elif len(name) <= 35:
		return "Medium"
	elif len(name) <= 45:
		return "Good"
	else:
		return "Long"

train_df["NameLength"] = train_df["Name"].map(get_name_length)
train_df = pd.get_dummies(train_df, columns = ["NameLength"])

test_df["NameLength"] = test_df["Name"].map(get_name_length)
test_df = pd.get_dummies(test_df, columns = ["NameLength"])

# fare group
def get_fare_group(fare):
	if fare <= 4:
		return "VeryLow"
	elif fare <= 10:
		return "Low"
	elif fare <= 20:
		return "Mid"
	elif fare <= 45:
		return "High"
	else:
		return "VeryHigh"

train_df["FareGroup"] = train_df["Fare"].map(get_fare_group)
train_df = pd.get_dummies(train_df, columns = ["FareGroup"])

test_df["FareGroup"] = test_df["Fare"].map(get_fare_group)
test_df = pd.get_dummies(test_df, columns = ["FareGroup"])

# age group
def get_age_group(age):
	if age <= 1:
		return "Infant"
	elif age <= 4:
		return "Toddler"
	elif age <= 13:
		return "Child"
	elif age <= 18:
		return "Teenager"
	elif age <= 35:
		return "YoungAdult"
	elif age <= 45:
		return "Adult"
	elif age <= 55:
		return "MiddleAged"
	elif age <= 65:
		return "Senior"
	else:
		return "Old"

train_df["AgeGroup"] = train_df["Age"].map(get_age_group)
train_df = pd.get_dummies(train_df, columns = ["AgeGroup"])

test_df["AgeGroup"] = test_df["Age"].map(get_age_group)
test_df = pd.get_dummies(test_df, columns = ["AgeGroup"])

# embarked
def get_embarked(embarked):
	if isinstance(embarked, float):
		return "N"
	if "Q" in embarked:
		return "Q"
	else:
		return "S"

train_df["EmbarkedGroup"] = train_df["Embarked"].map(get_embarked)
train_df = pd.get_dummies(train_df, columns = ["EmbarkedGroup"])

test_df["EmbarkedGroup"] = test_df["Embarked"].map(get_embarked)
test_df = pd.get_dummies(test_df, columns = ["EmbarkedGroup"])

# drop some features
Y = train_df["Survived"].values
train_df.drop(["Cabin", "Name", "Ticket", "Embarked", "Survived", "EmbarkedGroup_N", "Deck_T"], axis = 1, inplace = True)
test_df.drop(["Cabin", "Name", "Ticket", "Embarked"], axis = 1, inplace = True)
print(train_df.columns.values)
print(test_df.columns.values)

# print the processed data
pd.set_option('display.max_columns', None)
print(test_df)

# model
xgbc_pipeline = make_pipeline(Imputer(), XGBClassifier(max_depth = 3, n_estimators = 3000, learning_rate = 0.5))
# xgbc_pipeline.fit(train_df[train_df.columns.values], train_df["Survived"])

# lets start off with some predictions
seed = 0
X_train, X_test, y_train, y_test = train_test_split(train_df[train_df.columns.values], Y, test_size = 0.33, random_state=seed)
xgbc_pipeline.fit(X_train, y_train)
xgbc_preds = xgbc_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, xgbc_preds)
print(accuracy)
print(xgbc_preds)

xgbc_pipeline.fit(train_df, Y)
preds = xgbc_pipeline.predict(test_df[test_df.columns.values])

pred_df = pd.DataFrame(
		data={
			"PassengerId":test_df["PassengerId"].values,
			"Survived":preds
			}
		)
pred_df.to_csv("sub.csv", index = False)