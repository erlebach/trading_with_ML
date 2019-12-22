import matplotlib.pyplot as plt

def getDate(fn):
	s,y,m,d = scanf.scanf("%s_%4d%2d%2d", fn)
	return y,m,d

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

