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

# Starting from linear_regression_4.py
# Create a set of classes

# Starting from linear_regression-5.py (which produces incorrect results),, 
# fix all the results according to check_linear_regression.py . r2_score should
# not be near 1. That is impossible. 


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

class Trading:
    #---------------------------
    def __init__(self, folder, break_date, nb_train_days, nb_feature_days, wait_days, profit_thresh):
        self.folder = folder
        self.break_date = break_date
        self.nb_train_days = nb_train_days
        self.nb_feature_days = nb_feature_days
        self.wait_days = wait_days
        self.profit_thresh = profit_thresh
        pass

# This routine is different for each experiment
	def stock_func(self, sym='AAPL', n=1000):
	    # symbols/ contains 8000+ symbols from AMEX, NYSE, NASDAQ 
    	file = "symbols/" + sym + ".txt"
    	# number of symbols to keep
    	nb_keep = 1500  
    	df = pd.read_csv(file).iloc[-nb_keep:] 
    	df['x'] = df.reset_index().index.values # 0, 1, 2, ...
    	dates = df['date'].values
        return df[['x', 'date','c']].values

#----------------------------------------------------------------------
    #---------------------------
    def getTrainTest(self, stock_sym):
        # wait_days: how many days in the future to wait before measuring profit: [1,-]
        df = pd.read_csv(self.folder+stock_sym + ".txt", index_col=0)
        self.sym = stock_sym # <<<<<<

        break_date_record = df.loc[self.break_date]
        iloc_break_date = df.index.get_loc(self.break_date)

        # subtract nb_feature days to add to the training set, since each 
        # feature set is wait_days worth of data
        rec = df.iloc[iloc_break_date-self.nb_train_days-nb_feature_days]
        min_date = int(rec.name)

        # parenthesis and & are required for compound conditional
        train_df = df.loc[(df.index <= break_date) & (df.index >= min_date)]
        test_df  = df.loc[df.index  >  break_date]

        # training set based on close prices
        # labels are based on the close prices on the next day
    
        # Training features: close price, volume
        # Labels: tomorrow's price
        # Make sure that 'c' is first

        # Make sure that all features are in cols for the rest of the program to work properly!!

        #cols = ['c','o']   # cleans up code, and avoids repetition
        cols = ['c','o','l','h','vol']   # cleans up code, and avoids repetition
        cols = ['c','o','l','h']   # cleans up code, and avoids repetition
        cols = ['c', 'vol']   # cleans up code, and avoids repetition
        cols = ['c']   # cleans up code, and avoids repetition

        # reject the stock if the minimum 20-day average volume is less than 200,000 (over the testing
        # and training period)
        df['vol_sma_avg_20'] = df['vol'].rolling(window=20).mean()

        train_vol = df['vol_sma_avg_20'].loc[(df.index <= break_date) & (df.index >= min_date)]
        test_vol  = df['vol_sma_avg_20'].loc[df.index  >  break_date]

        min_avg_train_vol = train_vol.min()
        min_avg_test_vol  = test_vol.min()

        #print('min_avg_train_vol= ', min_avg_train_vol)
        #print('min_avg_test_vol= ',  min_avg_test_vol)

        if min_avg_train_vol < 200000 or min_avg_test_vol < 200000:
            return False

        train_df = train_df[cols]
        test_df  = test_df[cols]


        # days:  -2       -1          0
        #             yesterday     today
        # Given an array [0,1,2,3,..., n-2, n-1]
        # Assume that nb_feature_days = 3
        # wait_days = 3
        # Possible training sets: [0, 1, 2] -> 5, [1, 2, 3] -> 6, [n-4, n-3, n-2] -> n-3

        # Need a better way to set up the columns. 
        # Instead of extracting values, we will work with pandas

        # Step 1: Add columns to the dataframe with the additional required features 
        # Step 2: thin out the days at which the features are needed (optional)
        # Step 3: Take wait_days into account before extracting the data

        # Assume nb_feature_days is nb_feature_days
        # Final df cols:  day[0], day[-1], ..., day[1-nb_feature_days]
        # Each day: closing price and volume
        # cols:  'c0', 'vol0', 'c1', 'c2'
        # day[0]                 : rows[nb_feature_days-1 : -1]
        # day[1]                 : rows[nb_feature_days-2 : -2]
        # day[2]                 : rows[nb_feature_days-3 : -3]
        # day[nb_feature_days-1] : rows[0 : -nb_feature_days]

        train_day_d = []
        test_day_d  = []

        for i in range(0, nb_feature_days):
            j = nb_feature_days - 1 - i
            day = train_df.iloc[nb_feature_days-1-j : -1-j-wait_days]
            day = day.reset_index()  # 'date' becomes a column
            if i > 0: day.drop(['date'], inplace=True, axis=1)
            if i > 0: day.columns = [c + str(i) for c in day.columns]
            train_day_d.append( day )

            day = test_df.iloc[nb_feature_days-1-j : -1-j-wait_days]
            day = day.reset_index()  # 'date' becomes a column
            if i > 0: day.drop(['date'], inplace=True, axis=1)
            if i > 0: day.columns = [c + str(i) for c in day.columns]
            test_day_d.append( day )  

        # concat does a join on the index or specificed column. So reset_index was applied
        train_df = pd.concat(train_day_d, axis=1)
        test_df  = pd.concat(test_day_d,  axis=1)

        # Remove all date columns except the first (column 1)

        # Volume is sometimes zero. Must impute its value. Use mean value over 3 days prior and three days following. How to do this/ 
        # Closing prices are the first column

        self.x_train = train_df.values[0:-wait_days, 1:]  # do not include date
        self.y_train = train_df.loc[wait_days:,'c'].values # 

        self.x_test  = test_df.values[0:-wait_days, 1:] # 
        self.y_test  = test_df.loc[wait_days:,'c'].values # 

        self.train_dates = train_df.iloc[:-wait_days,0].values
        self.nb_train_days = self.train_dates.shape[0]
        self.test_dates = test_df.iloc[:-wait_days,0].values
        self.nb_test_days = self.test_dates.shape[0]

        n_features = self.x_train.shape[-1]

        """
        if (n_features == 1):
            print("cannot handle a single feature. Exit")
            #embed()
            return False
:y_train = train_df.loc[wait_days:,'c'].values
        self.train_dates = train_df.iloc[:-wait_days,0]
        self.nb_train_days = self.train_dates.shape[0]

        self.x_test  = test_df.values[0:-wait_days, 1:] # 200 test values, do not include date
        self.y_test  = test_df.loc[wait_days:,'c'].values # 200 test values
        self.test_dates = test_df.iloc[:-wait_days,0].values
        self.nb_test_days = self.test_dates.shape[0]

        n_features = self.x_train.shape[-1]
        """
    
        """
        if (n_features == 1):
            print("cannot handle a single feature. Exit")
            #embed()
            return False
        """

        self.feature_names = cols

        """
        print("\n\n")
        print("cols= ", cols)
        print("train_df")
        print(train_df.head())

        print("y_train: ", self.y_train.shape)   # one rows to few!!!! << ERROR ERROR ERROR
        print(self.y_train[0:5])
        print("x_train.shape: ", self.x_train.shape)
        print("y_train.shape: ", self.y_train.shape)
        print("train_dates.shape: ", self.train_dates.shape)

        print("\nself.train")
        print(self.x_train[0:5,:])
        print(self.y_train[0:5])
        print(self.train_dates[0:5])
        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.train_dates.shape)

        # Shape self.x_train: [nrows,ncols]
        # Shape self.y_train: [nrows,]

        print("\nself.test")
        print(self.x_test[0:5,:])
        print(self.y_test[0:5])
        print(self.test_dates[0:5])
        print(self.x_test.shape)
        print(self.y_test.shape)
        print(self.test_dates.shape)

        #quit()

        # MISSING: self.y_train and self.y_test
        """

        return True

    #----------------------------------
    def processStock(self):
        self.getTrainTest(self.sym) 
    
        # Create linear regression object
        regr = linear_model.LinearRegression()
    
        # What is the first column? The number of features? 
        # If so, I must do a transpose on the data
        #print("x_train, y_train shapes: ", self.x_train.shape, self.y_train.shape)
        regr.fit(self.x_train, self.y_train)
        #print("x_train: ", self.x_train.shape)
        #print("train_dates: ", self.train_dates[0:3])
    
        # Make predictions using the testing set
        #print("x_test.shape: ", self.x_test.shape)
        self.y_pred = regr.predict(self.x_test) #.reshape(-1)
        #quit()
    
        # The coefficients
        #print('Coefficients: \n', regr.coef_)
        #print('Intercept: \n', regr.intercept_)
    
        # The mean squared error
        # make sure that y_test and y_pred have same shape)
        # https://scikit-learn.org/stable/modules/feature_selection.html

        self.r2_score = r2_score(self.y_test, self.y_pred)

        #self.printMetrics()
        u.plotPredVsRealPrice(self.sym, self.x_test[:,0], self.y_test, self.y_pred)
        self.computeProfit()

    #---------------------------
    def computeProfit(self):
        real_profit = self.y_test - self.x_test[:,0]
        pred_profit = self.y_pred - self.x_test[:,0]
    
        # Plot outputs
        u.plotRealVsPredProfit(self.sym, real_profit, pred_profit)
    
        all_days_total_real_profit = real_profit.sum()
        all_days_total_pred_profit = pred_profit.sum()

        print("***** count all the days ****")
        print("total_real_profit= ", all_days_total_real_profit)
        print("total_pred_profit= ", all_days_total_pred_profit)
    
        # Only count profit on days where the predicted profit is positive

        profit = 100.*(pred_profit / self.x_test[:,0]) > profit_thresh
        real_profit = real_profit[profit]
        pred_profit = pred_profit[profit]
        profit_dates = self.test_dates[profit]
    
        print("shapes: x_test, real_profit, prd_profit, profit_dates")
        print(self.x_test.shape, real_profit.shape, pred_profit.shape, profit_dates.shape)
        print("test_dates: ", self.test_dates[0:5])
        print("Profit_shape: ", profit.shape) # 1D array
        print("real_profit= ", real_profit[0:5])
        print("pred_profit= ", pred_profit[0:5])
        print("profit dates= ", profit_dates[0:5])
    
        # Sum up the real profit over the test data
        total_real_profit = real_profit.sum()
        total_pred_profit = pred_profit.sum()
        print("***** count only the days where pred_profit > 0 ****")
        print("total_real_profit= ", total_real_profit)
        print("total_pred_profit= ", total_pred_profit)

        self.total_real_profit = total_real_profit
        self.total_pred_profit = total_pred_profit
        #quit()

    #---------------------------
    def printMetrics(self):
        
        print()
        print("=================================================")
        print("=========== METRICS =============================")
        print("Features (put in Pandas df): ", self.feature_names)
        print('Mean squared error: %.2f' % mean_squared_error(self.y_test, self.y_pred))
        print("Explained variance score: ", explained_variance_score(self.y_test, self.y_pred))

        mi = mutual_info_regression(self.x_train, self.y_train)
        mi = mi / np.max(mi)

        fr, pval = f_regression(self.x_train, self.y_train, center=True)  # center?

        print("Mutual Information: ", mi)
        #fr = fr / np.max(fr)
        print("f_regression: " , fr)
        print("pval: " , pval)
        print("R2 score: ", self.r2_score)
        print("=========== END METRICS =========================")
        print("=================================================")
        print()
    


#----------------------------------------------------------------------
stocks = ["AAPL", "ZEUS"]
stocks = ["ZEUS"]
stocks = ["ZBRA"]
stocks = ["SENEB"]
stocks = ["INDY"]
stocks = ["AAPL"]
stocks = ["AAPL", "INDY", "ZEUS", "ZBRA", "ZION", "YORW", "PEGA", "PCYO", "XOMA", "VIAB", "VECO", "WIRE", "MU", "AMD"]
#stocks = ["AAPL", "ZEUS", "ZBRA", "INDY"]

# Keep data up to 2018 for training
break_date = 20150701
break_date = 20160701
break_date = 20170705
folder = "symbols/"

# nb_train_days is decreased by wait_days
nb_train_days = [1, 25, 50, 100, 200, 400]
nb_train_days = [100, 500, 1000]
nb_train_days = [200]

# nb of days to use for the features
# the same features are used for each day
nb_feature_days_list = [1,5,10]
profit_thresh   = 0  # percentage profit below which I do not enter the trade
wait_days_list = [1,5,10,20,40] # how many days in the future to measure the profit 
wait_days_list = [1,5] # how many days in the future to measure the profit 


cols = ['sym', 'break_date', 'ndays', 
        'nb_feature_days', 'wait_days', 
        'profit_thresh', 'pred_profit', 'real_profit', 
        'nb_train_days', 'nb_test_days', 'r2_score']
df = pd.DataFrame(columns=cols)
records = []


for sym in stocks:
    for ndays in nb_train_days:
        #for wait_days in [1, 5, 10, 20]:
        for wait_days in wait_days_list:
            for nb_feature_days in nb_feature_days_list:
                # profit_thresh is in dollars
                print("---------------------------------------")
                print("wait_days: ", wait_days)
                print("nb_feature_days: ", nb_feature_days)
                print("nb_train_days: ", nb_train_days)
                trade = Trading(folder, break_date, ndays, nb_feature_days, wait_days, profit_thresh)
                ret = trade.getTrainTest(sym)
                if ret == False: 
                    break
                trade.processStock()
                records.append([sym, break_date, ndays, 
                                nb_feature_days, wait_days, 
                                profit_thresh, trade.total_pred_profit, trade.total_real_profit, 
                                trade.nb_train_days, trade.nb_test_days, trade.r2_score])

dfnew = pd.DataFrame(records, columns=cols) # 1st way
print(dfnew)
#df = pd.concat([df, dfnew])  # 2nd day, same result
#print(df)
