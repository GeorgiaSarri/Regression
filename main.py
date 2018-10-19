import pandas as pd
import time

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from modules import data_preroc
from modules import feature_selection
from modules import linear_regression
from modules import lasso, lassoCV
from modules import decisionTree, gradientBoost

#===========================================================
# Read dataset.
# The data are already split in  a train and a test set.
#===========================================================


sourceTrain = pd.read_csv('train.csv')
sourceTest = pd.read_csv('test.csv')

print(sourceTrain.shape)
print(sourceTest.shape)

dataTrain = data_preroc(sourceTrain)

# 1. Linear Regression

# Feature Selection - by using RFECV
features = feature_selection(dataTrain, LinearRegression())
print('Number of Original Features: ', len(dataTrain.columns)-1)
print('Number of Filtered Features: ', len(features))
print('Selected Features: ', features)
dataSelected = dataTrain[features]#.join(dataTrain.SalePrice)
trainX, testX, trainY, testY = train_test_split(dataSelected, dataTrain.SalePrice, test_size=0.1,
                                                random_state=42)
start_time = time.time()
lr_model, mse, score = linear_regression(trainX, trainY, testX, testY)
print('Linear Regression Execution time: ', (time.time() - start_time))
print('Linear Regression Score: ', score)
print('Mean square error: ', mse ** (0.5))

# - Observations :
# Ideally, the scatter plot should create a linear line, but since the model does not fit 100%,
# the scatter plot is not a line. Also the accuracy of our model is negative, meaning that this model
# fits our data poorly.

trainX, testX, trainY, testY = train_test_split(dataTrain.loc[:, dataTrain.columns != 'SalePrice']
                                                ,dataTrain.SalePrice
                                                , test_size=0.1
                                                ,random_state=42)

#2a. LASSO
start_time = time.time()
lasso_model, mse, score = lasso(trainX, trainY, testX, testY)
print('LASSO Execution time: ', (time.time() - start_time))
print('LASSO Score: ', score)
print('Mean square error: ', mse**(0.5))
print('LASSO alpha: ', lasso_model.get_params()['alpha'])
# - Observations :
# Ideally, the scatter plot should create a linear line. Since the model does not fit 100%,
# the scatter plot is not creating a linear line. Also the accuracy of our model is 92%.
# Improved MSE and we have feature selection by default and we do not need the extra step.

#2b. LASSO_CV
start_time = time.time()
lassoCV_model, mse, score = lassoCV(trainX, trainY, testX, testY)
print('LASSO_CV Execution time: ', (time.time() - start_time))
print('LASSO_CV Score: ', score)
print('LASSO_CV Mean square error: ', mse**(0.5))
print('LASSO_CV alpha: ', lassoCV_model.alpha_)
# - Observations :
# From the metrics the model performs worse than the other two,
# but it selects 13 features, which is easier to be explained.

#3. Decision Tree
start_time = time.time()
decisionTree_model, mse, score = decisionTree(trainX, trainY, testX, testY)
print('DecisionTree Execution time: ', (time.time() - start_time))
print('DecisionTree Score: ', score)
print('DecisionTree Mean square error: ', mse**(0.5))

#4. Gradient Boost
start_time = time.time()
gradBoost_model, mse, score = gradientBoost(trainX, trainY, testX, testY)
print('Gradient Boost Execution time: ', (time.time() - start_time))
print('Gradient Boost Score: ', score)
print('Gradient Boost Mean square error: ', mse**(0.5))
