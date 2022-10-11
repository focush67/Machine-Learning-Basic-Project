import pandas as pd
import numpy as np

housing = pd.read_csv("data.csv")

housing.head()
housing.info()

# housing['CHAS'].value_counts()
housing.describe()

# %matplotlib inline
# import matplotlib.pyplot as plt
# housing.hist(bins=50 , figsize = (20,15))
# plt.show()

# For implementation purpose only 
# def split_train_test(data , test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     print(shuffled)
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices] , data.iloc[test_indices]

# train_set , test_set = split_train_test(housing , 0.2)

from sklearn.model_selection import train_test_split
train_set , test_set = train_test_split(housing , test_size = 0.2 , random_state = 42)
print(f"ROWS IN TRAIN SET : {len(train_set)}\nROWS IN TEST SET : {len(test_set)}\n")

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1 , test_size = 0.2 , random_state = 42)
for train_index , test_index in split.split(housing , housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set.info()

strat_train_set.info()

housing = strat_train_set.copy()

corr_matrix = housing.corr()

corr_matrix['MEDV'].sort_values(ascending = False)

from pandas.plotting import scatter_matrix
attributes = ['RM' , 'ZN' , 'MEDV' , 'LSTAT']
scatter_matrix(housing[attributes] , figsize = (12,8))

# housing.plot(kind = "scatter" , x = "RM" , y = "MEDV", alpha = 0.8)

housing = strat_train_set.drop("MEDV" , axis = 1)
housing_labels = strat_train_set["MEDV"].copy()

# To take care of missing attriutes , we have 3 options 
#     1 : Get rid of missing data points
#     2 : Get rid of the whole attribute
#     3 : Set the value to some value(0,mean or median)

a = housing.dropna(subset = ["RM"]) # Option 1
a.shape

a = housing.drop("RM" , axis = 1)
a.shape# Option 2
# Note that there is no RM column and also the original housing data frame will repeat unchanged

median = housing["RM"].median() # Compute median for option 3
median

housing["RM"].fillna(median)
# Note that original data frame wil remain unchanged

housing.shape
# Before we started filling attributes

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)

imputer.statistics_
x = imputer.transform(housing)

housing_tr = pd.DataFrame(x , columns = housing.columns)
housing_tr.describe()

# # Primarily 3 types of ojects
#     1: Estimators : Estimates some parameters based on a dataset , eg- Imputer
#                     It has a fit() and transform()
#                     fit() - Fits the dataset and calculates internal parameters
                    
#     2 : Transformers : Takes input and returns output based on the learnings from fit().
#                        It also has a convenient function fit_transform() which fits and
#                        then transforms
                    
#     3 : Predictors : LinearRegression model is an example of predictor , fit() and predict()
#                      are 2 common function . It also gives score function which will evaluate
#                      predictions

# Primarily there are 2 types of feature scaling methods
#     1 : Min - Max Scaling (Normalization)
#         (value - min) / (max-min)
#         Sklearn provides class called MinMaxScaler for this
        
#     2 : Standardization
#         (value - mean) / standard deviation
#         Sklearn provides class called StandardScaler for this

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([('imputer' , SimpleImputer(strategy = "median")) , ("std_scaler" , StandardScaler())])

housing_num_tr = my_pipeline.fit_transform(housing)

housing_num_tr.shape

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
model = RandomForestRegressor()
model.fit(housing_num_tr , housing_labels)

some_labels = housing_labels.iloc[:5]
some_data = housing.iloc[:5]

prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)

list(some_labels)

from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels , housing_predictions)
rmse = np.sqrt(mse)

rmse

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model , housing_num_tr , housing_labels , scoring = "neg_mean_squared_error" , cv = 10)
rmse_scores = np.sqrt(-scores)

rmse_scores

def print_scores(scores):
    print("Scores are : ",scores)
    print("MEAN : ",scores.mean())
    print("Standard Deviation :",scores.std())

print_scores(rmse_scores)

from joblib import dump , load
dump(model , 'Singhania.jolib')

x_test = strat_test_set.drop("MEDV" , axis = 1)
y_test = strat_test_set["MEDV"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
final_mean_squared_error = mean_squared_error(y_test , final_predictions)
final_mse = mean_squared_error(y_test , final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions ,list(y_test))

final_rmse
