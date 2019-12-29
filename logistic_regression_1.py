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
import seaborn as sns
import utils as u
from IPython import embed

from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report


# Given the break date, and "n", compute min_date = break_date->getDays() - n
# So I need getDays given yyyymmdd day format (integer)
# The date is in the format integer yyyymmdd, month and date count from zero

#----------------------------------------------------------------------
# This routine is different for each experiment
def getTrainTest(stock_sym, break_date, nb_days):
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
    # labels are 1 if there is profit, otherwise zero

    # Training features: close price, volume
    # Labels: tomorrow's price
    cols = ['c','vol']   # cleans up code, and avoids repetition
    x_train = train_df[cols].values[0:-1] # returns numpy array [n,2]
    c = train_df['c'].values

    # False -> 0. True -> 1.
    # y_train has shape (n,)  (as opposed to (n)
    y_train = c[1:] - c[0:-1] > 0
    y_train = y_train.astype(float)

    x_test = test_df[cols].values[0:-1] # returns numpy array [n,2]
    c = test_df['c'].values
    y_test = c[1:] - c[:-1] > 0
    y_test = y_test.astype(float)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test
#----------------------------------------------------------------------

# break_date: separates training from label data
# nb_days: nb of days to use for the training data (could go from 1 to 250)
#    So I would use at most one year of training data

def processStock(stock_sym, break_date, nb_days, n_features):
    if (n_features == 1):
        print("cannot handle a single feature. Exit")
        embed()
        quit()

    x_train, y_train, x_test, y_test = getTrainTest(stock_sym, break_date, nb_days)

    # Create linear regression object
    # ml is more generic. Allows for generalization later on
    # Loop over some parameters
    #for c in [.001, .01, .1, 1,10,100,1000]: 
    for c in [100]:
        print("********* C = %d *********" % c)
        #ml = linear_model.LogisticRegression(C=c)  # What is C?
        #ml = DecisionTreeClassifier(criterion='entropy')  # What is C?
        ml = RandomForestClassifier(n_estimators=100, criterion='entropy')  # What is C?

        # If so, I must do a transpose on the data
        #print("x_train, y_train: ", x_train.shape, y_train.shape)
        print("x_train: ", x_train[0:30])
        print("y_train: ", y_train[0:30])
        ml.fit(x_train, y_train)

        # Make predictions using the testing set
        y_pred = ml.predict(x_test) #.reshape(-1)
        #  <<<<<< >>>>>>>
        # For AAPL and ZEUS, confusion matrix independent of C. Why? 
        #  <<<<<< >>>>>>>
        print("y_pred: ", y_pred.sum(), y_pred[0:30])
        print("y_test: ", y_test.sum(), y_test[0:30])

        # first argument must be true, 2nd the predicted values
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        quit()
    
    # The mean squared error
    # make sure that y_test and y_pred have same shape)
    print("x_test, y_test, y_pred: ", x_test.shape, y_test.shape, y_pred.shape)
    print("x_train, y_train: ", x_train.shape, y_train.shape)
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print("ALL CORRECT")

    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))


    # Strategy: On the days where pred_prof > 0, buy the stock. Sell the next day. So the actual profit
    # is real_profit

    pred_profit = y_pred
    print("pred_profit = ", pred_profit)
    profit = pred_profit > 0
    print("profit = ", profit.shape, profit)
    real_profit = real_profit[profit]
    print("real_profit.shape= ", real_profit.shape)

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

stocks = ["ZEUS", "AAPL"]
stocks = ["AAPL", "ZEUS"]

# Keep data up to 2018 for training
break_date = 20180000
folder = "symbols/"
nb_days = [1, 25, 50, 100, 200, 400]
nb_days = [500]   # business days, not trading days
# Another way of working with nb_days is to use iloc to find the starting
# date to use in training data. Then we are working with trading days. 
# Leave to future work. 

n_features = 2

for sym in stocks:
    for ndays in nb_days:
        print("---------------- %s, %d days -----------------" % (sym, ndays))
        # Might be better to input a list of features (string): ['c','vol']
        # So all features would be precomputed or be defined via function. 
        # Or the input could be a dictionary 'vol': getVolume, 'rsi':getRSI, ..
        processStock(folder + sym, break_date, -ndays, n_features)

