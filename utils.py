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

