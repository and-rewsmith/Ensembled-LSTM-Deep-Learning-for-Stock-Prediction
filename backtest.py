# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 00:55:54 2018

@author: Brandon
"""

import matplotlib.pyplot as plt
import numpy as np 

def backtest(predictions, dollars1, close_list, targets, ticker, split_value, sell_tolerance=0.5, buy_tolerance=0.5):

    close_list = close_list[::-1]
#    predictions = scaler.fit_transform(predictions)

    dollars = dollars1
    buy = 0
    buys = []
    sells = []
    shares = 0 
    account_balance = [dollars]
    cash = dollars
    win_rate = []
    
    for i in range(len(predictions)):
        
        rounded_preds = np.round(predictions)
        
        target = targets[i]
        
        if rounded_preds[i] - target == 0:
            win_rate.append(1)
        else:
            win_rate.append(0)
        
        if i%10 == 0:
            print(str(round((sum(win_rate)*100 / float(len(win_rate))),2)) + '% Win Rate through ' + str(i) + ' Simulated Trading Days')
        
    close_list = list(close_list)
    
    
    for i in range(len(predictions)):    
        
        if predictions[i] > buy_tolerance:
            if buy == 0:
                sells.append(0)
                buy = 1
                buy_value = close_list[i]
                buys.append(close_list[i])
                shares = account_balance[-1] // close_list[i]
                cash = round(dollars - (shares * close_list[i]),2)
                account_balance.append((shares*close_list[i])+ cash)
            else:
                buys.append(0)
                sells.append(0)
                account_balance.append((shares*close_list[i])+ cash)
        
        elif predictions[i] < sell_tolerance:
            
            if buy == 1:
                buys.append(0)
                sell_value = close_list[i]
                sells.append(close_list[i])
                profit = ((sell_value - buy_value) / buy_value) + 1
                dollars = profit * dollars
                buy = 0 
                cash+=round(shares*close_list[i],2)
                shares = 0 
                account_balance.append((shares*close_list[i])+ cash)
            else:
                sells.append(0)
                buys.append(0)
                account_balance.append((shares*close_list[i])+ cash)
        
        else:
            account_balance.append((shares*close_list[i])+ cash)

    ticker_balance = []
    shares = dollars1 / close_list[0]
    
    for i in range(len(predictions)):
        ticker_balance.append(shares*close_list[i])
        
    total_return = round((account_balance[-1]/dollars1) * 100,2)
    ticker_return = round((close_list[-1] / close_list[0])*100, 2)
    
    print('Algorithm achieved a return of ' + str(total_return) + '% in ' + str(len(account_balance)) + ' trading days')
    print(ticker + ' achieved a return of ' + str(ticker_return) + '% in this same period')
    
    algo_colors = []
    for i in range(len(account_balance)):
        if i < split_value:
            algo_colors.append('green')
        else:
            algo_colors.append('purple')
    
    #Calculating 1-Year Returns
    yr = 252
    year_ticker_balance = []
    
    ticker_initial = close_list[-yr]
    for close in close_list[-yr:]:
        year_ticker_balance.append(round(((close/ticker_initial) - 1)*100,2))
        
    year_balance_list = []
    initial_account_bal = account_balance[-yr]
    for bal in account_balance[-yr:]:
        year_balance_list.append(round(((bal/initial_account_bal) - 1)*100,2))
    
    plt.figure()
    plt.plot(account_balance, color = 'grey', label = 'Algo Results',linewidth=1.0)
    plt.scatter(range(len(account_balance)), account_balance, color = algo_colors, s = 1)
    plt.plot(ticker_balance, color = 'red', label = ticker + ' Results',linewidth=1.0)
    plt.title('Algorithm Results for ' + ticker)
    plt.xlabel('Days')
    plt.ylabel('Dollars')
    plt.legend(loc='upper left')
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    
    
    #Plotting 1 -yr return 
    plt.figure()
    plt.plot(year_balance_list, color = 'grey', label = 'Algo 1-Year Return', linewidth=1.0)
    plt.scatter(range(len(year_balance_list)), year_balance_list, color = algo_colors[-yr:], s = 3)
    plt.plot(year_ticker_balance, color = 'red', label = ticker + ' 1-Year Return', linewidth=1.0)
    plt.title('1-Year Return of Algo for ' + ticker)
    plt.xlabel('Trading days')
    plt.ylabel('Percent Gain/Loss')
    plt.legend(loc = 'upper left')
    # Customize the major grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
          
       