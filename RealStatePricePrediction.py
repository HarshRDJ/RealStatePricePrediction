"""
Real State Price Prediction based on input data.
Supervised Machine Learning
Language: Python
By Harsh Dewangan
Date: 10th April, 2021

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

housing = pd.read_csv('Data.csv')
# print(housing.head(10))
# print(housing.info())
# print(housing.describe())

# CHAS is a categorical variable
# print(housing['CHAS'].value_counts())

"""
# plot the variables to visualize them
housing.hist(bins=50, figsize=(20, 15))
plt.show()
"""

"""
# Train test split
def train_test_split(data, test_ratio):
  np.random.seed(25)
  shuffled = np.random.permutation(len(data))
  test_set_size = int(len(data) * test_ratio)
  test_indices = shuffled[:test_set_size]
  train_indices = shuffled[test_set_size:]
  return data.iloc[train_indices], data.iloc[test_indices]
"""

# independent and dependent feature
X = housing.iloc[:, :-1]
y = housing.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=25)
# print(f"Rows in train set: {len(X)}\nRows in test set: {len(y)}")

# StratifiedShuffleSplit is a cross-validation object that is a merge of StratifiedKFold and ShuffleSplit, which returns
# stratified randomized folds. The folds are made by preserving the percentage of samples for each class.
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=5, test_size=.2, random_state=25)
for train_index, test_index in split.split(housing, housing['CHAS']):
  strat_train_data = housing.loc[train_index]
  strat_test_data = housing.loc[test_index]
# print(f"Train set:\n{strat_train_data['CHAS'].value_counts()}")
# print(f"Test set:\n{strat_test_data['CHAS'].value_counts()}")

# we need to work with training data from here thus taking housing dataframe with only training sets
housing = strat_train_data.copy()

# Looking for correlation
corr_matrix = housing.corr()
# print(corr_matrix['MEDV'].sort_values(ascending=False))


from pandas.plotting import scatter_matrix
attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(housing[attributes], figsize=(15,10))
# plt.show()

housing.plot(kind='scatter', x="MEDV", y='LSTAT', alpha=.5)

# trying out attribute combinations
housing = strat_train_data.drop("MEDV", axis=1)
housing_label = strat_train_data["MEDV"].copy()

# Handling missing datas
median = housing['RM'].median()
housing['RM'].fillna(median)

# below task will automate the handling of missing data values in the dataset
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(housing)
I = imputer.transform(housing)
housing_tr = pd.DataFrame(I, columns=housing.columns)
# print(housing_tr.describe())

# Feature scaling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        # add as may as you need in you pipeline
                        ("std_scaler", StandardScaler()),])
housing_num_tr = my_pipeline.fit_transform(housing_tr)
# print(housing_num_tr)

# Setting a desired model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_label)
some_data = housing.iloc[:5]
some_label = housing_label.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
# print(model.predict(prepared_data))
# print(list(some_label))

# Evaluating the model
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_label, housing_predictions)
rmse = np.sqrt(mse)
# print(rmse)

# Using better validation technique - cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_label, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
# print(rmse_scores)

def print_scores(scores):
  print("Scores             :", scores)
  print("Mean               :", scores.mean())
  print("Standard Deviation :", scores.std())

#print_scores(rmse_scores)

from joblib import dump, load
dump(model, 'RealStatePricePrediction.joblib')

# Testing the model
X_test = strat_test_data.drop("MEDV", axis=1)
Y_test = strat_test_data["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_rmse)
#print(prepared_data[0])