# Regression
The objective is to complete a a full analysis of a Kaggle Dataset (https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
that concerns the prediction of a house's price based on various features.

The analysis was on pyhton and the goal was to familiriaze with python's sklearn module as well as compare different regression models.
The process starts with a simple data preprocessing were the null values of the dataset are handled and afterwards four regression models 
are tested.

Models used:
1) Linear Regression:
  After using recursive feature selection to choose the most important features a simple linear regression model was used to predict
  the houses' price.
2) Lasso:
  Both simple Lasso and Lasso with parameters optimization were used. Lasso proved ideal to reduse a large number of features down to 
  a set that could be explained in a model.
3) Decision Tree:
  After selecting the optimum values for max depth and min leaf samples a simple Decision Tree was tested.
4) Gradient Boost:
After selecting the optimum values for learning rate, max depth and min leaf samples Gradient Boost was tested. For this type of problem
this model is too "expensive" but its results were the best.


