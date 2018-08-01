# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:04:47 2018

@author: Brandon
"""

def addVariables(df, ticker):
    
    
    df = df.iloc[::-1]
    group = df['Close']
    long = group.ewm(span=26,adjust=False).mean()
    short = group.ewm(span=12,adjust=False).mean()
    macd = short-long
    signal = macd.ewm(span=9,adjust=False).mean()
    
    df['MACD'] = macd
    df['signal_line'] = signal
    
    MACD_signals = []
    for i in range(len(df)):
        day = df.iloc[i,:]
        
        if day['MACD'] > day['signal_line']:
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
            
    return df[::-1]
    
    
    