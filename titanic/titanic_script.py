import os
import pandas as pd
import numpy  as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer

train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

train_df.fillna(train_df.mean(), inplace = True)
test_df.fillna(test_df.mean(),   inplace = True)

# we are going to do a bit of feature engineering but first lets 
# keep track of all the features created
features = [
	"NameLength", "NameLengthSmall", "NameLengthMedium", "NameLengthLong", 
	"Title", "Rarity", "HasCabin", "Deck", "Age", "IsChild", "AgeGroup_Child",
	"AgeGroup_Infant", "AgeGroup_MiddleAged", "AgeGroup_Old", "AgeGroup_SeniorCitizen",
	"AgeGroup_Teenager", "AgeGroup_Toddler", "AgeGroup_YoungAdult", "FamilySize",
	"Single", "Fare", "SibSp", "Parch", "Sex", "FarePerPerson", "FareGroup_Low",
	"FareGroup_VeryLow", "FareGroup_Mid", "FareGroup_High", "FareGroup_VeryHigh"
]

# lets get name length
def get_name_len(df):
	name_len_list = []
	for name in df["Name"].values:
		name_len_list.append(len(name))
	return name_len_list

# now lets create new column for the name_len
train_df["NameLength"] = get_name_len(train_df)
test_df["NameLength"] = get_name_len(test_df)

# lets create dummy variables for name length
def name_small(size):
	if size <= 20:
		return 1
	return 0

def name_med(size):
	if size > 20 and size <= 35:
		return 1
	return 0

def name_long(size):
	if size > 35:
		return 1
	return 0

# get our dummy values
train_df["NameLengthSmall"]  = train_df["NameLength"].map(name_small)
train_df["NameLengthMedium"] = train_df["NameLength"].map(name_med)
train_df["NameLengthLong"]   = train_df["NameLength"].map(name_long)

test_df["NameLengthSmall"]  = test_df["NameLength"].map(name_small)
test_df["NameLengthMedium"] = test_df["NameLength"].map(name_med)
test_df["NameLengthLong"]   = test_df["NameLength"].map(name_long)

# lets extract the titles
titles = [
	"Mrs", "Mr", "Miss", "Master", "Dr", "Rev", "Major",
	"Col", "Mlle", "Ms", "Mme", "Don", "Sir", "the Countess",
	"Capt", "Jonkheer", "Lady"
]

def get_titles(df):
	title_list = []
	for name in df["Name"].values:
		for title in titles:
			if title in name:
				title_list.append(title)
				break
	return title_list 

train_df["Title"] = get_titles(train_df)
test_df["Title"] = get_titles(test_df)

# set some dummy variables for title feature
def get_rarity(title):
	if title == "Mr" or title == "Mrs" or title == "Miss" or title == "Master":
		return 1
	elif title == "Dr" or title == "Rev" or title == "Major" or title == "Col":
		return 2
	else:
		return 3

train_df["Rarity"] = train_df["Title"].map(get_rarity)
test_df["Rarity"]  = test_df["Title"].map(get_rarity)

def title_to_numeral(title):
	map = {"Mr":0, "Mrs":1, "Miss":2, "Master":3, "Dr":4, "Rev":5, "Major":6,
	"Col":7, "Mlle":8, "Ms":9, "Mme":10, "Don":11, "Sir":12, "the Countess":13,
	"Capt":14, "Jonkheer":15, "Lady":16}
	return map[title]

# process titles into numerals
train_df["Title"] = train_df["Title"].map(title_to_numeral)
test_df["Title"] = test_df["Title"].map(title_to_numeral) 

# lets do some feature engineering on the cabin
def get_has_cabin(df):
	has_cabin_list = []
	for cabin in df["Cabin"].values:
		if isinstance(cabin, float):
			has_cabin_list.append(0)
		else:
			has_cabin_list.append(1)
	return has_cabin_list

train_df["HasCabin"] = get_has_cabin(train_df)
test_df["HasCabin"]  = get_has_cabin(test_df)

deck_to_numeral = { 'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'T':7, 'G':8}
def get_deck_list(df):
	deck_list = []
	for cabin in df["Cabin"].values: 
		for deck in deck_to_numeral:
			if isinstance(cabin, float):
				deck_list.append(0)
				break;
			elif deck in cabin:
				deck_list.append(deck_to_numeral[deck])
				break;
	return deck_list

train_df["Deck"] = get_deck_list(train_df)
test_df["Deck"]  = get_deck_list(test_df)

# is child feature
def is_child(age):
	if age <= 16:
		return 1
	else:
		return 0

train_df["IsChild"] = train_df["Age"].map(is_child)
test_df["IsChild"] = test_df["Age"].map(is_child)

# lets create some dummy variables for the age group
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
		return "SeniorCitizen"
	else:
		return "Old"

train_df["AgeGroup"] = train_df["Age"].map(get_age_group)
train_df = pd.get_dummies(train_df, columns = ["AgeGroup"], drop_first = True)
test_df["AgeGroup"] = test_df["Age"].map(get_age_group)
test_df = pd.get_dummies(test_df, columns = ["AgeGroup"], drop_first = True)

# family size and other interesting linear features to consider
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

def is_single(fam):
	if fam <= 1:
		return 1
	else:
		return 0

train_df["Single"] = train_df["FamilySize"].map(is_single)
test_df["Single"]  = test_df["FamilySize"].map(is_single)

# lets create fare per person out of the family size
train_df["FarePerPerson"] = train_df["Fare"] / train_df["FamilySize"]
test_df["FarePerPerson"] = test_df["Fare"] / test_df["FamilySize"]

# lets also make fare groups as dummy variables
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
test_df["FareGroup"] = test_df["Fare"].map(get_fare_group)
train_df = pd.get_dummies(train_df, columns = ["FareGroup"], drop_first = False)
test_df = pd.get_dummies(test_df, columns = ["FareGroup"], drop_first = False)

# map 1 and 0 for sex
def get_sex(sex):
	if sex == "male":
		return 1
	else:
		return 0

train_df["Sex"] = train_df["Sex"].map(get_sex)
test_df["Sex"] = test_df["Sex"].map(get_sex)

print(train_df["Embarked"].values)
# lets start off with some predictions
#seed = 0
#X_train, X_test, y_train, y_test = train_test_split(train_df[features], train_df["Survived"], test_size = 0.33, random_state=seed)
xgbc_pipeline = make_pipeline(Imputer(), XGBClassifier())
#xgbc_pipeline.fit(X_train, y_train)
#xgbc_preds = xgbc_pipeline.predict(X_test)
#accuracy = accuracy_score(y_test, xgbc_preds)
#print(accuracy)
#print(xgbc_preds)

# now lets check using k fold cross validation
# scores = cross_val_score(xgbc_pipeline, train_df[features], train_df["Survived"], cv = 5)
# print(scores)

# ok lets now compile real predictions into csv
xgbc_pipeline.fit(train_df[features], train_df["Survived"])
preds = xgbc_pipeline.predict(test_df[features])
pred_df = pd.DataFrame(
		data={
			"PassengerId":test_df["PassengerId"].values,
			"Survived":preds
			}
		)
pred_df.to_csv("sub.csv", index = False)
print("sub.csv")