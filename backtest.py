# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 00:55:54 2018

@author: Brandon
"""

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np 

def backtest(df, model, dollars1, close_list, ticker, sell_tolerance=0.5, buy_tolerance=0.5):
    dollars = dollars1
    df = df[::-1]
    close_list = close_list[::-1]
    
    targets = list(df['Next_Change'])
    
    train = df.drop('Next_Change', axis = 1)
    
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    predictions = model.predict(train) 
    
#    predictions = scaler.fit_transform(predictions)

    
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
        
    close_list = list(close_list.iloc[20:])
    for i in range(len(df)):    
        
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

    algo_balance = []
    shares = dollars1 / close_list[0]
    for i in range(len(df)):
        algo_balance.append(shares*close_list[i])
        
    
    plt.figure()
    plt.plot(account_balance, color = 'green', label = 'Algo Results')
    plt.plot(algo_balance, color = 'red', label = ticker + ' Results')
    plt.title('Algorithm Results for ' + ticker)
    plt.xlabel('Days')
    plt.ylabel('Dollars')
    plt.legend(loc='upper right')
    plt.show()
    
          
       