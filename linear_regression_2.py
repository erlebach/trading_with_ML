# Author: Gordon Erlebacher

# Linear_regression_2.py
# Only difference with linear_regression_1.py is that we will sue 
# profits as a lable (close today - close yesterday), and actual profit 
# as a label (close tomorrow - close today)

# starting with a single stock with symbol SYM, the training data will be
# (closing price today, closing price tomorrow). 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as u

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Keep data up to 2018 for training

folder = "symbols/"
stock = "AAPL"
stock = "ZEUS"

# Using 1-day profit to predict is not the way to go. 

#----------------------------------------------------------------------
def getTrainingLabelsTest(df):
    # parenthesis and & are required for compound conditional
    break_date = 20180000
    min_date   = 20100000
    train_df = df.loc[(df.index <= break_date) & (df.index >= min_date)]
    test_df  = df.loc[df.index  >  break_date]

    # training set based on one-day profit
    # Could experiment with "n-day profit". Create a function for this. 
    # labels are based on the close prices on the next day

    x_train = train_df.c.values[1:-1] - train_df.c.values[0:-2]
    y_train = train_df.c.values[2:]   - train_df.c.values[1:-1]

    print(train_df.c.values[1:-1].shape, train_df.c.values[0:-2].shape)
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)

    x_test = test_df.c.values[1:-1] - test_df.c.values[0:-2]
    y_test = test_df.c.values[2:]   - test_df.c.values[1:-1]

    return x_train, y_train, x_test, y_test
#----------------------------------------------------------------------


def processStock(stock_sym, model):
    df = pd.read_csv(stock_sym + ".txt", index_col=0)

    # First, we will try a linear regression on a single stock using sklearn

    x_train, y_train, x_test, y_test = getTrainingLabelsTest(df)

    # I now have my training and labels as 1D arrays


    # Train the model
    # Need 2D arrays. The reshape is necessary if there is only a single feature (2nd dimension)
    model.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))

    # Make predictions using the testing set
    y_pred = model.predict(x_test.reshape(-1,1)).reshape(-1)

    # The coefficients
	# These only exist for linear regression. How to generalize
    print('Coefficients: \n', model.coef_)
    print('Intercept: \n', model.intercept_)

    # The mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

    # The coefficient of determination: 1 is perfect prediction
    # Might only work for linear regression (not sure)
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

# Create linear regression object
model = linear_model.LinearRegression()
processStock(folder + stock, model)
