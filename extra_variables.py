# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:04:47 2018

@author: Brandon
"""

import pandas as pd
import numpy as np

def addVariables(df, ticker):
    df = df[::-1]
    
    group = df['Close']
    long = group.ewm(span=26,adjust=False).mean()
    short = group.ewm(span=12,adjust=False).mean()
    macd = short-long
    signal = macd.ewm(span=9,adjust=False).mean()
    
    df['MACD'] = macd
    df['signal_line'] = signal
    MACD_signals = []
    for i in range(len(df)):
        
        if i <= 1:
            MACD_signals.append(0)
        
        else:
            day = df.iloc[i,:]
            previous = df.iloc[i-1,:]
            previous_two = df.iloc[i-2,:]
            
        
            if day['MACD'] > previous['MACD'] and previous['MACD'] < previous_two['MACD']:
                MACD_signals.append(1)
            else:
                if day['MACD'] > previous['MACD'] and MACD_signals[-1] == 1:
                    MACD_signals.append(1)
                else:
                    MACD_signals.append(0)
    
    df['MACD_signals'] = MACD_signals

    previous_MACD_signal = []
    MACD_change = []
    signal_change = []
    
    for i in range(len(df)):
        if i == 0:
            previous_MACD_signal.append(0)
            MACD_change.append(0)
            signal_change.append(0)
        else:
            day = df.iloc[i,:]
            previous = df.iloc[i-1,:]
            
            if day['MACD'] > previous['MACD']:
                MACD_change.append(1)
            else:
                MACD_change.append(0)
                
            if day['signal_line'] > previous['signal_line']:
                signal_change.append(1)
            else:
                signal_change.append(0)
                
            previous_MACD_signal.append(previous['MACD_signals'])
    
    convergence = []
    
    for i in range(len(df)):
        if i == 0:
            convergence.append(0)
        else:
            day = df.iloc[i,:]
            previous = df.iloc[i-1,:]
            
            current_MACD_distance = day['MACD'] - day['signal_line']
            previous_MACD_distance = previous['MACD'] - previous['signal_line']
            
            convergence.append(current_MACD_distance - previous_MACD_distance)
        
    df['previous_MACD_signal'] = previous_MACD_signal        
    df['MACD_change'] = MACD_change  
    df['signal_change'] = signal_change  
    df['convergence'] = convergence
    df = df[::-1]      
    return df

def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    u.ewm(com=period-1,adjust=False).mean()
    rs = u.ewm(com=period-1,adjust=False).mean() / d.ewm(com=period-1,adjust=False).mean()
    return 100 - 100 / (1 + rs)

def add_RSI(df):
    
    df = df[::-1]
    df['RSI'] = RSI(df['Close'], 14)    
    df = df[::-1]
    return df
    
    
    