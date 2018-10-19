import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# ==========================================================
# data_preroc
# - input:
#       dataTrain : Train Dataset from file
#       dataTest  : Test Dataset from file
# - description:
#   Data Preprocessing
#   Remove redundant columns
#   NAs handling
#   Categorical Features Handling
# ==========================================================
def data_preroc(data):

    data = data.drop('Id', axis = 1)

    #From the variables shown below Alley, PoolQC, Fence and MiscFeature
    # have too many NAs to be considered important so we delete them.
    cols = []
    freq = []
    for x in data.columns.values:
        if data[x].isnull().sum() > 0:
            cols.append(x)
            freq.append(data[x].isnull().sum())

    indices = np.arange(len(cols))
    plt.bar(indices, freq, color='r')
    plt.xticks(indices, cols, rotation='vertical')
    plt.show()

    cols = [ 'Alley', 'PoolQC', 'Fence' ,'MiscFeature']
    data = data.drop(cols, axis = 1)
    #print(data.columns.values)

    # For the MasVnrType, MasVnrArea, BsmtQual, BsmtExposure,
    # BsmtFinType1, BsmtFinType2, GarageType, GarageYrBlt, GarageFinish,
    # GarageQual and GarageCond we see that they do not have
    # many NAs so we can drop the corresponding lines.
    cols = ['MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType'
            ,'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'Electrical']

    for x in cols:
        data = data.dropna(subset = [x])
    #print(data.shape)

    # For the rest of the columns having null
    # values fill the NAs with the mean (if nuerical) or most frequent value(if categorical).
    # From the below we get that only two columns remain with NAs, one nuerical and one categorical

    cols = []
    freq = []
    for x in data.columns.values:
        if data[x].isnull().sum() > 0:
            cols.append(x)
            freq.append(data[x].isnull().sum())

    indices = np.arange(len(cols))
    plt.bar(indices, freq, color='r')
    plt.xticks(indices, cols, rotation='vertical')
    plt.show()

    print("Mean Value: ", data.LotFrontage.mean())
    print("Count of Na: ",data.LotFrontage.isnull().sum())
    print("Values\n ",data.FireplaceQu.value_counts())
    print("Count of Na: ",data.FireplaceQu.isnull().sum())

    #We will fill the LotFrontage na with its mean value = 70.76
    # and the FireplaceQu with its two dominant value: Gd.
    data['LotFrontage'] = data['LotFrontage'].fillna(70.76)
    data['FireplaceQu'] = data['FireplaceQu'].fillna('Gd')

    #Convert categorical to dummy variables
    data = pd.get_dummies(data)
    #print(data.dtypes)
    #print('------------------------------')
    #print('Categorical Variables: ', data.select_dtypes(include=[object]).columns.values)

    return data

#====================================================================
#
# Feature Selection
#
#====================================================================
def feature_selection(data, estimator):
    trainY = data.SalePrice
    trainX = data.loc[:, data.columns != 'SalePrice']  # pd.get_dummies()

    rfecv = RFECV(estimator=estimator, scoring='neg_mean_squared_error', cv=3) #mse returned by CV
    rfecv = rfecv.fit(trainX, trainY)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    return trainX.columns[rfecv.support_]

#====================================================================
# Data Analyis; Use/Compare different models to predict house prices
 # Linear Regression
 # LASSO
 # Decision Trees
 # Gradient Boost
#====================================================================

# 1. Linear Regression
# -- Linear regression is the simplest and most widely used statistical technique for predictive modeling
def linear_regression(trainX, trainY, testX, testY):
    lr = LinearRegression()
    model = lr.fit(trainX, trainY)
    Y_pred = np.round(model.predict(testX))
    # remove outliers, we could remove them completely
    # but instead make the error equal to the true price.
    Y_pred[np.where(np.abs(Y_pred)>10**9)] = 0

    plt.scatter(testY, Y_pred)
    plt.xlabel("Prices: $Y_i$")
    plt.ylabel("Predicted prices: $\hat{Y}_i$")
    plt.title("Linear Regression: Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
    plt.show()

    return model, mean_squared_error(testY, Y_pred), model.score(testX, testY)

#2a. Lasso
def lasso(trainX, trainY, testX, testY, alpha = 1.0, max_iter = 1000):
    las = Lasso(alpha=alpha, max_iter=max_iter)
    model = las.fit(trainX, trainY)

    Y_pred = model.predict(testX)
    Y_pred[np.where(np.abs(Y_pred)>10**9)] = 0 #remove outliers

    plt.scatter(testY, Y_pred)
    plt.xlabel("Prices: $Y_i$")
    plt.ylabel("Predicted prices: $\hat{Y}_i$")
    plt.title("LASSO: Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
    plt.show()
    print('LASSO Features Excluded: ',  trainX.columns[np.where(model.coef_ == 0)])

    return model, mean_squared_error(testY, Y_pred), model.score(testX, testY)

#2b. LassoCV
def lassoCV(trainX, trainY, testX, testY, max_iter=3000, cv=5, n_threads=3):
   lasCV = LassoCV(max_iter=max_iter, cv=cv, n_jobs=n_threads)
   #sfm = SelectFromModel(lasCV, threshold=0.25)
   model = lasCV.fit(trainX, trainY)

   #features = sfm.transform(trainX)

   Y_pred = model.predict(testX)
   Y_pred[np.where(np.abs(Y_pred) > 10 ** 9)] = 0  # remove outliers

   plt.scatter(testY, Y_pred)
   plt.xlabel("Prices: $Y_i$")
   plt.ylabel("Predicted prices: $\hat{Y}_i$")
   plt.title("LASSO_CV: Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
   plt.show()
   print('LASSO_CV: Features Selected: ', trainX.columns[np.where(model.coef_ != 0)])

   return model, mean_squared_error(testY, Y_pred), model.score(testX, testY)

#3. Decision Trees
def decisionTree(trainX, trainY, testX, testY, cv=5):
    # FInd best Parameters for our tree regressor.
    parameters = [
        {'max_depth': list(range(1, 15)),
         'min_samples_leaf': list(range(2, 20))}
    ]
    clf = GridSearchCV(DecisionTreeRegressor(), parameters, cv=cv)

    clf = clf.fit(trainX, trainY)
    print('decisionTree: Best Estimator: ', clf.best_estimator_)
    print('decisionTree: Best Parameters: ', clf.best_params_)

    max_depth = clf.best_params_['max_depth']
    min_samples_leaf = clf.best_params_['min_samples_leaf']

    dtr = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    model = dtr.fit(trainX, trainY)
    Y_pred = model.predict(testX)

    plt.scatter(testY, Y_pred)
    plt.xlabel("Prices: $Y_i$")
    plt.ylabel("Predicted prices: $\hat{Y}_i$")
    plt.title("Decision Tree - Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

    return model, mean_squared_error(testY, Y_pred), model.score(testX, testY)

#4. Gradient Boost
# Good explanation of GB: http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/ (kudos!!!)
def gradientBoost(trainX, trainY, testX, testY, cv=5, n_jobs = 2):
    #the more the values in the param_grid the more time it takes.
    param_grid = {'learning_rate': [0.1, 0.05],  # , 0.02, 0.01],
                  'max_depth': [6, 4],  # ,6],
                  'min_samples_leaf': [3, 5, 9, 17],
                  }
    cbr = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=cv, n_jobs=n_jobs)
    model = cbr.fit(trainX, trainY)
    Y_pred = model.predict(testX)

    print('GradientBoost: Best Estimator: ', cbr.best_estimator_)
    print('GradientBoost: Best Parameters: ', cbr.best_params_)

    plt.scatter(testY, Y_pred)
    plt.xlabel("Prices: $Y_i$")
    plt.ylabel("Predicted prices: $\hat{Y}_i$")
    plt.title("Gradient Boost - Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

    return model, mean_squared_error(testY, Y_pred), model.score(testX, testY)
