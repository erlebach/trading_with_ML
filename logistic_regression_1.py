# Author: Gordon Erlebacher

# Linear_regression_1.py
# Create functions and moved plotting to utils.py

# We demonstrate the use of random forest to predict a single stock. 
# Given the price of today, predict the price of tomorrow. 

# starting with a single stock with symbol SYM, the training data will be
# (closing price today, closing price tomorrow). 

# Training set: price today
# Label set: price tomorrow 

# Starting from linear_regression_1.py
# Instead of considering 10 years of data, consider "n days" of data, 
# with "n" being an input variable. 
# Alternatively (not done yet), I could choose n days of data randomly over 
# a certain interval. Say, choose randomly 50 dates across a two year period. 
# In the future, each choice would by m days of data previous to "today". 
# Again, this is not coded.

# Starting from linear_regression_4.py
# Implement random forest, SVM and logistic regression
# training data: price + vol. 
# label: is there a profit the next day

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as u
from IPython import embed

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Given the break date, and "n", compute min_date = break_date->getDays() - n
# So I need getDays given yyyymmdd day format (integer)
# The date is in the format integer yyyymmdd, month and date count from zero

#----------------------------------------------------------------------


# break_date: separates training from label data
# nb_days: nb of days to use for the training data (could go from 1 to 250)
#    So I would use at most one year of training data

def processStock(stock_sym, break_date, nb_days, n_features):
    if (n_features == 1):
        print("cannot handle a single feature. Exit")
        embed()
        quit()

    df = pd.read_csv(stock_sym + ".txt", index_col=0)

    # First, we will try a linear regression on a single stock using sklearn

    # parenthesis and & are required for compound conditional
    min_date = u.addDays(break_date, nb_days)
    print("break_date= ", break_date)
    print("min_date= ", min_date)
    print("nb_days= ", nb_days)

    train_df = df.loc[(df.index <= break_date) & (df.index >= min_date)]
    test_df  = df.loc[df.index  >  break_date]

    # training set based on close prices
    # labels are based on the close prices on the next day

    cols = ['c','vol']   # cleans up code, and avoids repetition
    x_train = train_df[cols].values[0:-1]
    # Profit. Use a condition, converted to a float (0., 1.). Scalar. 
    y_train = ((train_df['c'].values[1:] - train_df['c'].values[0:-1]) > 0).astype(np.float)
    print("x_train: ", x_train.shape)
    print("y_train: ", x_train.shape)

    x_test = test_df[cols].values[0:-1]
    y_test = ((test_df['c'].values[1:] - test_df['c'].values[0:-1]) > 0).astype(np.float)

    # Create linear regression object
    #regr = linear_model.LinearRegression()
    regr = linear_model.LogisticRegression(C=1e5)  # What is C?

    # Train the model
    # Need 2D arrays. The reshape is necessary if there is only a single feature (2nd dimension)
    #regr.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))

    # *** The features are in the last slot of the arrays / tensors

    # Example with Logistic function: 
    # https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html#sphx-glr-auto-examples-linear-model-plot-logistic-py

    # What is the first column? The number of features? 
    # If so, I must do a transpose on the data
    print("x_train, y_train: ", x_train.shape, y_train.shape)
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    print("x_test: ", x_test.shape)
    y_pred = regr.predict(x_test) #.reshape(-1)

    # The coefficients (NOT SURE WHAT THESE MEAN for logistic regression)
    print('Coefficients: \n', regr.coef_)
    print('Intercept: \n', regr.intercept_)

    # The mean squared error
    # make sure that y_test and y_pred have same shape)
    print("y_test, y_pred: ", y_test.shape, y_pred.shape)
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))

    #print("x_test, ", x_test.shape)
    #print("y_test, ", y_test.shape)
    #print("y_pred, ", y_pred.shape)

    real_profit = y_test - x_test   # <<< ERROR
    quit()
    pred_profit = y_pred - x_test

    # Strategy: On the days where pred_prof > 0, buy the stock. Sell the next day. So the actual profit
    # is real_profit

    profit = pred_profit > 0
    real_profit = real_profit[profit]

    print("x_test= ", x_test[0:10])
    print("y_test= ", y_test[0:10])
    print("y_pred= ", y_pred[0:10])
    print("real_profit= ", real_profit[0:10])
    print("pred_profit= ", pred_profit[0:10])

    # Sum up the real profit over the test data
    total_real_profit = real_profit.sum()
    total_pred_profit = pred_profit.sum()
    print("total_real_profit= ", total_real_profit)
    print("total_pred_profit= ", total_pred_profit)

stocks = ["AAPL", "ZEUS"]
stocks = ["ZEUS", "AAPL"]

# Keep data up to 2018 for training
break_date = 20180000
folder = "symbols/"
nb_days = [1, 25, 50, 100, 200, 400]
nb_days = [50]
n_features = 2

for sym in stocks:
    for ndays in nb_days:
        print("---------------- %s, %d days -----------------" % (sym, ndays))
        # Might be better to input a list of features (string): ['c','vol']
		# So all features would be precomputed or be defined via function. 
		# Or the input could be a dictionary 'vol': getVolume, 'rsi':getRSI, ..
        processStock(folder + sym, break_date, -ndays, n_features)

