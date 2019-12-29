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

# Starting from linear_regression_3.py
# Add volume as a second feature

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as u
from IPython import embed

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import f_regression, mutual_info_regression, f_regression


# Given the break date, and "n", compute min_date = break_date->getDays() - n
# So I need getDays given yyyymmdd day format (integer)
# The date is in the format integer yyyymmdd, month and date count from zero

#----------------------------------------------------------------------
# This routine is different for each experiment
def getTrainTest(stock_sym, break_date, nb_train_days):
    df = pd.read_csv(stock_sym + ".txt", index_col=0)

    # First, we will try a linear regression on a single stock using sklearn

    # parenthesis and & are required for compound conditional
    min_date = u.addDays(break_date, nb_train_days)
    print("break_date= ", break_date)
    print("min_date= ", min_date)
    print("nb_train_days= ", nb_train_days)

    train_df = df.loc[(df.index <= break_date) & (df.index >= min_date)]
    test_df  = df.loc[df.index  >  break_date]

    # nb of days to use for the features
    # the same features are used for each day
    nb_feature_days = 5

    # training set based on close prices
    # labels are based on the close prices on the next day

    # Training features: close price, volume
    # Labels: tomorrow's price
    # Make sure that 'c' is first
    #cols = ['c','vol']   # cleans up code, and avoids repetition
    cols = ['c','o']   # cleans up code, and avoids repetition
    cols = ['c','o','l','h']   # cleans up code, and avoids repetition
    cols = ['c','o','l','h','vol']   # cleans up code, and avoids repetition
    cols = ['c']   # cleans up code, and avoids repetition

    x_train = train_df[cols].values[0:-1]

    # days:  -2       -1          0
    #             yesterday     today
    # Given an array [0,1,2,3,..., n-2, n-1]
    # Assume that nb_feature_days = 3
    # Possible training sets: [0, 1, 2] -> 3, [1, 2, 3] -> 4, [n-4, n-3, n-2] -> n-1

    # Formula valid for any value of feature_days
    lst = []
    for n in range(nb_feature_days):  # [0,1,2]
        lst.append(train_df[cols].values[n:-nb_feature_days+n])
    x_train = np.hstack(lst) # stack along columns

    """
    # Manual creation of formula
    x_train_1 = train_df[cols].values[0:-3]
    x_train_2 = train_df[cols].values[1:-2]
    x_train_3 = train_df[cols].values[2:-1]
    x_train = np.hstack([x_train_1, x_train_2, x_train_3])  # stack along columns
    print(x_train.shape)
	"""
    

    # Predict closing price the next day
    y_train = train_df['c'].values[nb_feature_days:]

    lst = []
    for n in range(nb_feature_days):
        lst.append(test_df[cols].values[n:-nb_feature_days+n])
    x_test = np.hstack(lst) # stack along columns
    y_test = test_df['c'].values[nb_feature_days:]

    n_features = x_train.shape[-1]

    if (n_features == 1):
        print("cannot handle a single feature. Exit")
        embed()
        quit()

    feature_names = cols
    target = 'price'

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test, feature_names
#----------------------------------------------------------------------

# break_date: separates training from label data
# nb_train_days: nb of days to use for the training data (could go from 1 to 250) (pos or neg)
#    So I would use at most one year of training data

def processStock(stock_sym, break_date, nb_train_days):
    x_train, y_train, x_test, y_test, feature_names = getTrainTest(stock_sym, break_date, nb_train_days)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # What is the first column? The number of features? 
    # If so, I must do a transpose on the data
    #print("x_train, y_train: ", x_train.shape, y_train.shape)
    #print("x_test, y_test: ", x_test.shape, y_test.shape)
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    #print("x_test: ", x_test.shape)
    y_pred = regr.predict(x_test) #.reshape(-1)


    # The coefficients
    print('Coefficients: \n', regr.coef_)
    print('Intercept: \n', regr.intercept_)

    # The mean squared error
    # make sure that y_test and y_pred have same shape)
    # https://scikit-learn.org/stable/modules/feature_selection.html
    #print("x_test, y_test, y_pred: ", x_test.shape, y_test.shape, y_pred.shape)
    print()
    print("=================================================")
    print("=========== METRICS =============================")
    print("Features (put in Pandas df): ", feature_names)
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print("Explained variance score: ", explained_variance_score(y_test, y_pred))
    mi = mutual_info_regression(x_train, y_train)
    mi = mi / np.max(mi)
    print("Mutual Information: ", mi)
    fr, pval = f_regression(x_train, y_train, center=True)  # center?
    #fr = fr / np.max(fr)
    print("f_regression: " , fr)
    print("pval: " , pval)
    print("R2 score: ", r2_score(y_test, y_pred))
    print("=========== END METRICS =========================")
    print("=================================================")
    print()

    u.plotPredVsRealPrice(stock_sym, x_test[:,0], y_test, y_pred)

    print("x_test.shape= ", x_test.shape)
    print("y_test.shape= ", y_test.shape)

    real_profit = y_test - x_test[:,0]
    print("real_profit.shape: ", real_profit.shape)
    pred_profit = y_pred - x_test[:,0]
    print("pred_profit.shape: ", pred_profit.shape)

    # Plot outputs
    u.plotRealVsPredProfit(stock_sym, real_profit, pred_profit)

    total_real_profit = real_profit.sum()
    total_pred_profit = pred_profit.sum()
    print("***** count all the days ****")
    print("total_real_profit= ", total_real_profit)
    print("total_pred_profit= ", total_pred_profit)

    # Only count profit on days where the predicted profit is positive
    profit = pred_profit > 0
    real_profit = real_profit[profit]
    pred_profit = pred_profit[profit]
    print("real_profit.shape: ", real_profit.shape)
    print("pred_profit.shape: ", pred_profit.shape)

    print("x_test= ", x_test[0:10])
    print("y_test= ", y_test[0:10])
    print("y_pred= ", y_pred[0:10])
    print("real_profit= ", real_profit[0:10])
    print("pred_profit= ", pred_profit[0:10])

    # Sum up the real profit over the test data
    total_real_profit = real_profit.sum()
    total_pred_profit = pred_profit.sum()
    print("***** count only the days where pred_profit > 0 ****")
    print("total_real_profit= ", total_real_profit)
    print("total_pred_profit= ", total_pred_profit)

#----------------------------------------------------------------------
stocks = ["AAPL", "ZEUS"]
stocks = ["INDY"]
stocks = ["ZEUS"]
stocks = ["AAPL"]
stocks = ["ZBRA"]

# Keep data up to 2018 for training
break_date = 20180000
folder = "symbols/"
nb_train_days = [1, 25, 50, 100, 200, 400]
nb_train_days = [1000]

for sym in stocks:
    for ndays in nb_train_days:
        print("---------------- %s, %d days -----------------" % (sym, ndays))
        processStock(folder + sym, break_date, -ndays)

