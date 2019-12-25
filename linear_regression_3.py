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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as u

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Given the break date, and "n", compute min_date = break_date->getDays() - n
# So I need getDays given yyyymmdd day format (integer)
# The date is in the format integer yyyymmdd, month and date count from zero

#----------------------------------------------------------------------


# break_date: separates training from label data
# nb_days: nb of days to use for the training data (could go from 1 to 250)
#    So I would use at most one year of training data

def processStock(stock_sym, break_date, nb_days):
    df = pd.read_csv(stock_sym + ".txt", index_col=0)

    # First, we will try a linear regression on a single stock using sklearn

    # parenthesis and & are required for compound conditional
    min_date = u.addDays(break_date, -30)
    print("break_date= ", break_date)
    print("min_date= ", min_date)
    train_df = df.loc[(df.index <= break_date) & (df.index >= min_date)]
    test_df  = df.loc[df.index  >  break_date]

    # training set based on close prices
    # labels are based on the close prices on the next day

    x_train = train_df.c.values[0:-1]
    y_train = train_df.c.values[1:]
    #print(x_train.shape, x_train.shape)
    #print(x_train[0:10], y_train[0:10])

    x_test = test_df.c.values[0:-1]
    y_test = test_df.c.values[1:]
    #print(y_test.shape, y_test.shape)

    # I now have my training and labels as 1D arrays

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model
    # Need 2D arrays. The reshape is necessary if there is only a single feature (2nd dimension)
    regr.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))

    # Make predictions using the testing set
    y_pred = regr.predict(x_test.reshape(-1,1)).reshape(-1)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    print('Intercept: \n', regr.intercept_)

    # The mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))


    u.plotPredVsRealPrice(stock_sym, x_test, y_test, y_pred)


    # The plot looks nice and linear (for AAPL). However, how does that translate to profit. Assume
    # that that Monday, before the market opens, I run this code and find that te stock will go up
    # y is today's price at close, and x is yesterday's closing price. If y > x, I assume the stock 
    # will go up. So I purchase the stock, and sell it the next day. That is a profit of (y-x) per share. 
    # I am neglecting transaction costs, slippage, and latency (the order is not executed the moment it is 
    # sent. So I am estimating an idea profit. So on days that I predict the market goes up, I purchase 100 
    # shares, and sell at close to make or lose money. Let us see how lineargression does. 

    #print("x_test, ", x_test.shape)
    #print("y_test, ", y_test.shape)
    #print("y_pred, ", y_pred.shape)
    real_profit = y_test - x_test
    pred_profit = y_pred - x_test

    #print(real_profit.shape, pred_profit.shape)

    # Plot outputs
    # If predictions are correct, I'd expect all the points to be in the 1st and 4th quadrants. 
    # That is not the case. 
    u.plotRealVsPredProfit(stock_sym, real_profit, pred_profit)


    # Strategy: On the days where pred_prof > 0, buy the stock. Sell the next day. So the actual profit
    # is real_profit

    profit = pred_profit > 0
    real_profit = real_profit[profit]

    print("x_test= ", x_test[0:10])
    print("y_test= ", y_test[0:10])
    print("y_pred= ", y_pred[0:10])
    print("real_profit= ", real_profit[0:10])
    print("pred_profit= ", pred_profit[0:10])

    # The predicted profit is always positive. This makes sense, since the regression line is always
    # has a positive slope. So the price is always increasing. Even in down-years. A better approach 
    # would be to do linear regression of the last "n" days and measure profit as a function of "n". 

    # Sum up the real profit over the test data
    total_real_profit = real_profit.sum()
    total_pred_profit = pred_profit.sum()
    print("total_real_profit= ", total_real_profit)
    print("total_pred_profit= ", total_pred_profit)
    # For AAPL, the total real profit is 4x larger than total pred profit. That is strange. 
    #   I do not understand how this is possible. The real profit is pos and neg and the pred
    #   is always positive. Therefore, the real profit should cancel itself out. 
    #   The reason is that the predicted profit is very small. The real profit is 10x larger and skewed
    #    towards positive numbers, so the sum of real profits is still much higher. 
    # For ZEUS, the total real profit is 1.3, total pred_profit is 12. Opposite from AAPL. The key


    # Next will be to add a feature: say, the volume. And experiment with whether the volume should be 
    # normalized or not. 

    # Next, add the Open, Low and High to the linaer regression and see what happens. 
    
    # It looks like the scatter of actual to expected profit is isotropic, 
    # which implies very little correlation between predicted and expected loss. 
    # This suggests that we should perhaps train on the profit rather than the price. 
    # We will try that later. 

stocks = ["AAPL", "ZEUS"]

# Keep data up to 2018 for training
break_date = 20180000
folder = "symbols/"
nb_days = [1, 25, 50, 100, 200, 400]

for sym in stocks:
    for nd in nb_days:
        print("------------------ ", sym, nd, " -------------------")
        processStock(folder + sym, break_date, nd)
