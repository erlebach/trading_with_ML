import matplotlib.pyplot as plt
from datetime import date
from datetime import datetime, timedelta

#--------------------------------------------
def diffDates(date1, date2):
    # return (date2-date1) as a number of days 
    d1 = day2Datetime(date1)
    d2 = day2Datetime(date2)
    return date2-date1.days

def addDays(date1, nb_days):
    # this can lead to weekends, not necessarily trading days
    d1 = day2Datetime(date1)
    d2 = d1 + timedelta(days=nb_days)
    d2 = d2.strftime("%Y%m%d")
    print("d2= ", d2)
    d2 = int(d2) - 101   # d2 counts days and months from zero
    return d2

def day2Datetime(date):
    return datetime.strptime(str(date+101), "%Y%m%d")

def getDate(fn):
    s,y,m,d = scanf.scanf("%s_%4d%2d%2d", fn)
    return y,m,d
#--------------------------------------------

def plotPredVsRealPrice(sym, x_test, y_test, y_pred):
    # Plot outputs
    # Plot the price tomorrow on the vertical axis and today's price on the horizontal axis. 
    plt.scatter(x_test, y_test,  color='black')
    plt.plot(x_test, y_pred, color='blue', linewidth=3)
    plt.xlabel("x_test")
    plt.ylabel("y_test")
    #plt.xticks(())
    #plt.yticks(())
    plt.savefig(sym + "_price_today_tomorrow.pdf")

def plotRealVsPredProfit(sym, real_profit, pred_profit):
    plt.scatter(real_profit, pred_profit,  color='black')
    plt.xlabel("real_profit")
    plt.ylabel("pred_profit")
    plt.savefig(sym + "_real_pred_profit.pdf")
#----------------------------------------------------------------------
# Plot of stock characteristics
def plotStockData(trade):
    plt.subplots(3,3)
    plt.subplot(3,3,1)
    plt.scatter(trade.x_train[:,0], trade.x_train[:,1], s=.1)
    plt.title("%s, train: today vs yesterday" % trade.sym, fontsize=6)
    plt.xlabel("close")
    plt.ylabel("close")

    plt.subplot(3,3,2)
    plt.scatter(trade.x_train[:,0], trade.y_train[:], s=.1)
    plt.title("%s, today vs label" % trade.sym, fontsize=8)
    plt.xlabel("close")
    plt.ylabel("close")

    plt.subplot(3,3,3)
    plt.scatter(trade.x_test[:,0], trade.y_pred[:], s=.1)
    plt.title("%s, today vs label" % trade.sym, fontsize=8)
    plt.xlabel("close")
    plt.ylabel("close")

    plt.subplot(3,3,4)
    plt.scatter(trade.train_index[:], trade.x_train[:,0], s=.1)
    plt.title("Training")
    plt.ylabel("close")
    plt.xlabel("date")

    plt.subplot(3,3,5)
    plt.scatter(trade.test_index[:], trade.x_test[:,0], s=.1)
    plt.title("Testing")
    plt.ylabel("close")
    plt.xlabel("date")

    plt.subplot(3,3,6)
    plt.scatter(trade.y_test, trade.y_pred, s=.1)
    plt.title("test pred vs test labels", fontsize=8)
    plt.xlabel("pred close")
    plt.ylabel("test close")

    real_profit = trade.y_test[:] - trade.x_test[:,0]
    pred_profit = trade.y_pred[:] - trade.x_test[:,0]

    print("y_test: ", trade.y_test[0:10])
    print("x_test: ", trade.x_test[0:10])
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
    plt.savefig("plot_" + trade.sym + ".pdf")
    print("nb_train: ", trade.x_train.shape[0])
    print("nb_test: ", trade.x_test.shape[0])
    print("r2_score= ", trade.r2_score)

#----------------------------------------------------------------------
