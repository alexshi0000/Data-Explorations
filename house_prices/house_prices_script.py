import os
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# import data
train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')

# preprocess and impute missing values
test_id  = test_df['Id'].values
test_df  = test_df.drop(['Id'], 1) # now drop ids
train_df = train_df.drop(['Id'], 1)

# drop sale price after getting y
y_train  = train_df['SalePrice'].values
train_df = train_df.drop(['SalePrice'], 1)

# great looks like there are no features that are exclusive

# use vector operations to find missing ratio of all features
train_missing_data = (train_df.isnull().sum() / len(train_df)) * 100
train_missing_data = train_missing_data.sort_values(ascending = False)

test_missing_data = (test_df.isnull().sum() / len(test_df)) * 100
test_missing_data = test_missing_data.sort_values(ascending = False)

# print(train_missing_data)
# print(test_missing_data)

''' these are the ones to deal with
PoolQC           99.520548
MiscFeature      96.301370
Alley            93.767123
Fence            80.753425
FireplaceQu      47.260274
LotFrontage      17.739726
GarageCond        5.547945
GarageType        5.547945
GarageYrBlt       5.547945
GarageFinish      5.547945
GarageQual        5.547945
BsmtExposure      2.602740
BsmtFinType2      2.602740
BsmtCond          2.534247
BsmtQual          2.534247
BsmtFinType1      2.534247
MasVnrArea        0.547945
MasVnrType        0.547945
Electrical        0.068493
'''

# for the features below, nan means no pool is there, so this is a feature
train_df['PoolQC']      = train_df['PoolQC'].fillna('None')
train_df['MiscFeature'] = train_df['MiscFeature'].fillna('None')
train_df['Alley']       = train_df['Alley'].fillna('None')
train_df['Fence']       = train_df['Fence'].fillna('None')
train_df['FireplaceQu'] = train_df['FireplaceQu'].fillna('None')

test_df['PoolQC']      = test_df['PoolQC'].fillna('None')
test_df['MiscFeature'] = test_df['MiscFeature'].fillna('None')
test_df['Alley']       = test_df['Alley'].fillna('None')
test_df['Fence']       = test_df['Fence'].fillna('None')
test_df['FireplaceQu'] = test_df['FireplaceQu'].fillna('None')

# for LotFrontage the area is going to be similar to the different houses in the
# same neighborhood, so for each neighborhood fill the nans in the nbhood with mean

# lets get a set of neighborhoods
def get_lotfrontage(df):
	visited = []
	for nbhood in df['Neighborhood'].values:
		if nbhood in visited:
			continue
		visited.append(nbhood)
		df['LotFrontage'][df['Neighborhood'] == nbhood] =\
			df['LotFrontage'][df['Neighborhood'] == nbhood].fillna(\
			df['LotFrontage'][df['Neighborhood'] == nbhood].mean())
	return df

train_df = get_lotfrontage(train_df)
test_df  = get_lotfrontage(test_df)





# no garage cond means no garage to begin with
for col in ['GarageCond', 'GarageType', 'GarageFinish', 'GarageQual']:
	train_df[col] = train_df[col].fillna('None')
	test_df[col] = test_df[col].fillna('None')

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
	train_df[col] = train_df[col].fillna(0)
	test_df[col] = test_df[col].fillna(0)

# set na for bsmt
for col in ['BsmtExposure', 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 'BsmtFinType1']:
	train_df[col] = train_df[col].fillna('None')
	test_df[col] = test_df[col].fillna('None')

for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF',\
'BsmtFullBath', 'BsmtHalfBath']:
	train_df[col] = train_df[col].fillna('None')
	test_df[col] = test_df[col].fillna('None')

train_df['MasVnrType'] = train_df['MasVnrType'].fillna('None')
test_df['MasVnrType'] = test_df['MasVnrType'].fillna('None')

train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)
test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(0)

train_df['MSZoning'] = train_df['MSZoning'].fillna(train_df['MSZoning'].mode()[0])
test_df['MSZoning'] = test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])

train_df["Functional"] = train_df["Functional"].fillna("Typ")
test_df["Functional"] = test_df["Functional"].fillna("Typ")

train_df['Electrical'] = train_df['Electrical'].fillna(train_df['Electrical'].mode()[0])
test_df['Electrical'] = test_df['Electrical'].fillna(test_df['Electrical'].mode()[0])

train_df['KitchenQual'] = train_df['KitchenQual'].fillna(train_df['KitchenQual'].mode()[0])
test_df['KitchenQual'] = test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])

train_df['Exterior1st'] = train_df['Exterior1st'].fillna(train_df['Exterior1st'].mode()[0])
test_df['Exterior2nd'] = test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])

train_df['SaleType'] = train_df['SaleType'].fillna(train_df['SaleType'].mode()[0])
test_df['SaleType'] = test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])

train_df['MSSubClass'] = train_df['MSSubClass'].fillna("None")
test_df['MSSubClass'] = test_df['MSSubClass'].fillna("None")

# turn some discrete features into categorical data
# MSSubClass=The building class
train_df['MSSubClass'] = train_df['MSSubClass'].apply(str)
test_df['MSSubClass'] = test_df['MSSubClass'].apply(str)

# Changing OverallCond into a categorical variable
train_df['OverallCond'] = train_df['OverallCond'].astype(str)
test_df['OverallCond'] = test_df['OverallCond'].astype(str)

# Year and month sold are transformed into categorical features.
train_df['YrSold'] = train_df['YrSold'].astype(str)
test_df['YrSold'] = test_df['YrSold'].astype(str)

train_df['MoSold'] = train_df['MoSold'].astype(str)
test_df['MoSold'] = test_df['MoSold'].astype(str)

# overall quality is only out of 8, so we can treat that as
# catagorical data
train_df['OverallQual'] = train_df['OverallQual'].astype(str)
test_df['OverallQual'] = test_df['OverallQual'].astype(str)


# one hot encoding lets get our dummy variables
dummies_train = train_df.select_dtypes(object)
dummies_test = test_df.select_dtypes(object)

train_df = pd.get_dummies(train_df, columns = dummies_train)
test_df = pd.get_dummies(test_df, columns = dummies_test)


for col in train_df.columns.values:
	if not col in test_df.columns.values:
		train_df = train_df.drop(col, 1)

for col in test_df.columns.values:
	if not col in train_df.columns.values:
		test_df = test_df.drop(col, 1)

train_df.to_csv("the_data.csv")

# lets do some stack regression modelling now
X_train = train_df
X_test  = test_df

def rmse(y_true, y_pred):
	return np.sqrt(mean_squared_error(y_true, y_pred))


gbr_pipeline = make_pipeline(Imputer(),\
GradientBoostingRegressor(n_estimators = 7200, learning_rate = 0.05, max_depth = 6))

gbr_pipeline.fit(X_train, y_train)
gbr_pred = gbr_pipeline.predict(X_test)



ensemble = gbr_pred

output = pd.DataFrame(
    data = {
        "Id":test_id, "SalePrice":ensemble
    }
) 
# no indexing
output.to_csv("sub.csv", index = False)

print("done")

