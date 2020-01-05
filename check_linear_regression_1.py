# test of linear regression with known functions
# Follow approach of linear_regression_5.py
# Author: Gordon Erlebacher


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as u
import IPython
from IPython import embed

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import f_regression, mutual_info_regression, f_regression

import seaborn as sb

#-----------------------
def func(n=5000):
    x = np.linspace(0., n, n).reshape(-1,1)  # make into 2D array: one column)
    x /= n
    y = x**2
    # added noise
    y = np.sin(200*x)*np.exp(-3*x) + np.cos(180*x)*np.exp(.3*x) + 0.00*np.random.random(n).reshape(-1,1)
    #y1 = 0.*np.sin(20*x)*np.exp(-3*x)

    # Random numbers
    #y = 10.*np.random.random(n).reshape(-1,1)
    print("y: ", y.shape)
    #print("y= ", y[0:10])
    #print("y1= ", y1[0:10]); #quit()
    return np.hstack([x, y])

#-----------------------
def stock_func(sym='AAPL', n=1000):
    file = "symbols/" + sym + ".txt"
    # number of symbols to keep
    nb_keep = 1500  
    df = pd.read_csv(file).iloc[-nb_keep:] 
    df['x'] = df.reset_index().index.values # 0, 1, 2, ...
    dates = df['date'].values
    # Make sure 'c' is the first features listed after 'date'
    #return df[['x', 'date','c', 'vol']].values
    return df[['x', 'date','c']].values

#----------------------------------------------------------------------
class Trading:
    #---------------------------
    def __init__(self, folder, nb_train_days, nb_feature_days, wait_days, profit_thresh):
        self.folder = folder
        self.nb_train_days = nb_train_days
        self.nb_feature_days = nb_feature_days
        self.wait_days = wait_days
        self.profit_thresh = profit_thresh

    #---------------------------
    def getTrainTest(self, stock_sym):
        self.sym = stock_sym # <<<<<<
        n = 1000
        data = func(n)

        # return data columns (x, data, closing price) + a column of dates (yyyymmdd)
        data = stock_func(stock_sym, n)
        nb_features_per_day = data.shape[1] - 2  # 'c', or ['c', 'vol'], etc.

        train_day_d = []
        test_day_d  = []
        label = []
        nb_feature_days = self.nb_feature_days
        wait_days = self.wait_days

        print("nb_feature_days= ",  self.nb_feature_days)
        print("wait_days= ",  wait_days)

        for i in range(0, nb_feature_days):
            # I want current date to always be the first column
            # extract x (0,1,...n) and date columns for the current training day
            if (i == 0): 
                x     = data[nb_feature_days-1-i : -i-wait_days , 0]
                dates = data[nb_feature_days-1-i : -i-wait_days , 1]
            day = data[nb_feature_days-1-i : -i-wait_days , 2:]
            #print("day.shape: ", day[0:3,:]); quit()
            train_day_d.append( day )

        print("x: ", x.shape)
        print("x= ", x); 
        print("data.shape= ", data[0:2].shape); 
        print("data= ", data[0:2]); 
        label = data[nb_feature_days-1+wait_days :, -nb_features_per_day ]
        print("label shape: ", label.shape)
        print("label : ", label)

        # One column for each of nb_feature_days features
        print("data.shape: " , len(train_day_d), train_day_d[0].shape); 
        #data = np.asarray(train_day_d).T  
        data = np.hstack(train_day_d)
        #print("label: ", label); quit()
        data = np.hstack([x.reshape(-1,1), dates.reshape(-1,1), data, label.reshape(-1,1)])
        print("data.shape: " , data.shape); 
        print("data: " , data[0,:]); 
        #print("data: " , data[1,:]); quit()
        #data = np.hstack((x.reshape(-1,1), dates.reshape(-1,1), data, label.reshape(-1,1)))  # all args of hstack have same # dimensions

        print("data= ", data.shape)
        print("x= ", x.shape)
        print("label= ", label.shape)
        print("data= \n", data)
        
        print("before hstack")
        print("data= ", data.shape)
        print("data= ", data.shape)
        print("label= ", label.shape)
        # prefix "x" column, postfix "label" column
        nfpd = nb_features_per_day
        #print(data.shape); quit()
        #data = np.hstack((x.reshape(-1,1), dates.reshape(-1,1), data, label.reshape(-1,1)))  # all args of hstack have same # dimensions
        print("after hstack")
        print("data= ", data.shape)
        print("data= ", data.shape)
        # Data columns: x (0), dates (1), features (2), label (3)

        # Separate "data" into training and testing datasets
        nb_train = int(0.7 * label.shape[0])
        train_data = data[0:nb_train]
        test_data  = data[nb_train:]
        #print(train_data[0]); quit()
        train, train_label = train_data[:, 2:-1], train_data[:, -1]
        test, test_label   = test_data[:, 2:-1],  test_data[:, -1]

        print("train_data: ", train_data.shape)
        print("test_data: ", test_data.shape)
        print("train, train_label: ", train.shape, train_label.shape)
        print("test, test_label: ", test.shape, test_label.shape)
        print("Train:\n", train[0:5,:])  # contains only dates
        print("Train label:\n", train_label[0:5])  # contains closing prices
        print("Test:\n", test[0:5,:])  # contains only dates
        print("Test label:\n", test_label[0:5])  # contains closing prices
        #print(test, test_label)
        #quit()

        self.x_train = train
        self.y_train = train_label
        self.train_dates = train_data[:,1] # real scalars
        self.train_index = train_data[:,0] # real scalars

        self.x_test  = test
        self.y_test  = test_label
        self.test_dates = test_data[:,1]  # real scalars
        self.test_index = test_data[:,0]  # real scalars

        print("train_dates: ", self.train_dates)
        print("train_index: ", self.train_index)
        print("test_dates: ", self.test_dates)
        print("test_index: ", self.test_index)
        #quit()

        n_features = self.x_train.shape[-1] 

        return True

    #----------------------------------
    def processStock(self):
        self.getTrainTest(self.sym) 
    
        # Create linear regression object
        regr = linear_model.LinearRegression()
        regr.fit(self.x_train, self.y_train)

        # Make predictions using the testing set
        self.y_pred = regr.predict(self.x_test) #.reshape(-1)
        self.r2_score = r2_score(self.y_test, self.y_pred)
        print("r2_score= ", self.r2_score)

        plt.subplots(3,3)
        plt.subplot(3,3,1)
        plt.scatter(self.x_train[:,0], self.x_train[:,1], s=.1)
        plt.title("%s, train: today vs yesterday" % self.sym, fontsize=6)
        plt.xlabel("close")
        plt.ylabel("close")

        plt.subplot(3,3,2)
        plt.scatter(self.x_train[:,0], self.y_train[:], s=.1)
        plt.title("%s, today vs label" % self.sym, fontsize=8)
        plt.xlabel("close")
        plt.ylabel("close")

        plt.subplot(3,3,3)
        plt.scatter(self.x_test[:,0], self.y_pred[:], s=.1)
        plt.title("%s, today vs label" % self.sym, fontsize=8)
        plt.xlabel("close")
        plt.ylabel("close")

        plt.subplot(3,3,4)
        plt.scatter(self.train_index[:], self.x_train[:,0], s=.1)
        plt.title("Training")
        plt.ylabel("close")
        plt.xlabel("date")

        plt.subplot(3,3,5)
        plt.scatter(self.test_index[:], self.x_test[:,0], s=.1)
        plt.title("Testing")
        plt.ylabel("close")
        plt.xlabel("date")

        plt.subplot(3,3,6)
        plt.scatter(self.y_test, self.y_pred, s=.1)
        plt.title("test pred vs test labels", fontsize=8)
        plt.xlabel("pred close")
        plt.ylabel("test close")

        real_profit = self.y_test[:] - self.x_test[:,0]
        pred_profit = self.y_pred[:] - self.x_test[:,0]

        print("y_test: ", self.y_test[0:10])
        print("x_test: ", self.x_test[0:10])
        print("real_profit: ", real_profit[0:10])
        print("pred_profit: ", pred_profit[0:10])
        print("real_profit min/max: ", real_profit.min(), real_profit.max())
        print("pred_profit min/max: ", pred_profit.min(), pred_profit.max())

        plt.subplot(3,3,7)
        plt.scatter(real_profit, pred_profit, s=.1)
        plt.hlines(0, real_profit.min(), real_profit.max(), colors='red', lw=1)
        plt.vlines(0, pred_profit.min(), pred_profit.max(), colors='red', lw=1)
        plt.title("pred vs real profit", fontsize=8)
        plt.ylabel("pred profit")
        plt.xlabel("real profit")

        plt.tight_layout()
        plt.savefig("plot.pdf")
        print("nb_train: ", self.x_train.shape[0])
        print("nb_test: ", self.x_test.shape[0])
        print("r2_score= ", self.r2_score)
        quit()
    
#----------------------------------------------------------------------
stocks = ["ZBRA"]
stocks = ["IBM"]
stocks = ["AAPL"]
stocks = ["DDD"]
stocks = ["MU"]

# Keep data up to 2018 for training

# nb_train_days is decreased by wait_days
nb_train_days = [1, 25, 50, 100, 200, 400]
nb_train_days = [100, 500, 1000]
nb_train_days = [200]

# nb of days to use for the features
# the same features are used for each day
nb_feature_days_list = [4,5,10]
profit_thresh   = 0  # percentage profit below which I do not enter the trade
wait_days_list = [10] # how many days in the future to measure the profit 

cols = ['sym', 'break_date', 'ndays', 
        'nb_feature_days', 'wait_days', 
        'profit_thresh', 'pred_profit', 'real_profit', 
        'nb_train_days', 'nb_test_days', 'r2_score']
df = pd.DataFrame(columns=cols)
records = []
folder = "/"

for sym in stocks:
    for n_train_days in nb_train_days:
        #for wait_days in [1, 5, 10, 20]:
        for wait_days in wait_days_list:
            for nb_feature_days in nb_feature_days_list:
                # profit_thresh is in dollars
                print("---------------------------------------")
                #print("wait_days: ", wait_days)
                #print("nb_train_days: ", nb_train_days)
                trade = Trading(folder, n_train_days, nb_feature_days, wait_days, profit_thresh)
                ret = trade.getTrainTest(sym)
                if ret == False: 
                    #print("return False")
                    break
                trade.processStock()
                #records.append([sym, break_date, n_train_days, wait_days, 
                                #profit_thresh, trade.total_pred_profit, trade.total_real_profit, 
                                #trade.nb_train_days, trade.nb_test_days, trade.r2_score])

dfnew = pd.DataFrame(records, columns=cols) # 1st way
print(dfnew)
#df = pd.concat([df, dfnew])  # 2nd day, same result
#print(df)
