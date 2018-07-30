# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from mpl_finance import candlestick_ohlc
import os.path
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split

#Turning off Pandas iloc warning
pd.options.mode.chained_assignment = None  # default='warn'

def pandas_candlestick_ohlc(dat, stick = "day", otherseries = None):
    """
    :param dat: pandas DataFrame object with datetime64 index, and float columns "Open", "High", "Low", and "Close", likely created via DataReader from "yahoo"
    :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs include "day", "week", "month", and "year", ("day" default), and any numeric input indicates the number of trading days included in a period
    :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines
 
    This will show a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
    """
    
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the day
    
    # Create a new DataFrame which includes OHLC data for each period specified by stick input
    transdat = dat.loc[:,["Open", "High", "Low", "Close"]]
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1 # Used for plotting
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1]) # Identify weeks
            elif stick == "month":
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month) # Identify months
            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0]) # Identify years
            grouped = transdat.groupby(list(set(["year",stick]))) # Group by year and other appropriate variable
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                            "High": max(group.High),
                                            "Low": min(group.Low),
                                            "Close": group.iloc[-1,3]},
                                           index = [group.index[0]]))
            if stick == "week": stick = 5
            elif stick == "month": stick = 30
            elif stick == "year": stick = 365
 
    elif (type(stick) == int and stick >= 1):
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        grouped = transdat.groupby("stick")
        plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                        "High": max(group.High),
                                        "Low": min(group.Low),
                                        "Close": group.iloc[-1,3]},
                                       index = [group.index[0]]))
 
    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')
 
 
    # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots(figsize=(20,20))
    fig.subplots_adjust(bottom=0.2)
    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)
 
    ax.grid(True)
 
    # Create the candelstick chart
    candlestick_ohlc(ax, list(zip(list(date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
                      plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                      colorup = "black", colordown = "red", width = stick * .4)
 
    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:,otherseries].plot(ax = ax, lw = 1.3, grid = True)
 
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()


def get_data(ticker):
    
    key_list = []
    with open('keys.txt') as keys:
        lines = keys.readlines()
        for line in lines:
            key_list.append(line)      
    av_key = key_list[0]
    ts = TimeSeries(av_key, retries=100)
    data, metadata = ts.get_daily(ticker, outputsize='full')
    data = pd.DataFrame.from_dict(data)
    df = data.T
    df.columns = ['Open','High','Low','Close','Volume']
    df.index = pd.to_datetime(df.index)
    for column in df.columns :
        df[column] = pd.to_numeric(df[column])
    next_day_change=[]
    
    for i in range(len(df)):
        if i == 0:
            pass
        else:
            day = df.iloc[i,:]    
            next_day = df.iloc[i-1,:]
            next_change = next_day['Close'] - day['Close']
            if next_change >= 0:
                next_day_change.append(1)
            else:
                next_day_change.append(0)
    df = df.iloc[1:,:]
    df['Next_Change'] = next_day_change
    return df

def dataVariables(df, ticker):
    
    columns = list(df.columns)[0:-1]
    Open_1d_change = []
    High_1d_change = []
    Low_1d_change = []
    Close_1d_change = []
    Volume_1d_change = []
    
    for i in range(len(df)-1):
        day = df.iloc[i,:]    
        next_day = df.iloc[i+1,:]
        
        for col in columns:
            data = (next_day[col] - day[col])/day[col]
            if col == 'Open':
                Open_1d_change.append(data)
            if col == 'High':
                High_1d_change.append(data)
            if col == 'Low':
                Low_1d_change.append(data)
            if col == 'Close':
                Close_1d_change.append(data)
            if col == 'Volume':
                Volume_1d_change.append(data)
                
    df = df.iloc[0:-1:]
    
    df['Open_1d_change'] = Open_1d_change
    df['High_1d_change'] = High_1d_change
    df['Low_1d_change'] = Low_1d_change
    df['Close_1d_change'] = Close_1d_change
    df['Volume_1d_change'] = Volume_1d_change
    df.to_csv(ticker + '_data.csv')
    
    return df


def model(df):
    
    model = Sequential()
    num_inputs = len(list(df.columns)[0:-1])
    model.add(Dense(250, input_dim=num_inputs))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(0.15))
    model.add(Activation('relu'))
    model.add(Dense(20))
    model.add(Dropout(0.025))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model 

def model_train(df, model):
    
    y = df['Next_Change']
    X = df.drop('Next_Change', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model.fit(X_train, y_train, epochs=500, verbose=1, validation_data=(X_test,y_test))
    
if __name__ == '__main__':
    
    ticker = 'SPY'
    
    if os.path.isfile(ticker):
        df = pd.read_csv(ticker + '_data.csv')
    else:
        df = get_data(ticker)  
        df = dataVariables(df, ticker)
        
    df = df.drop('Open', axis=1)
    df = df.drop('High', axis=1)
    df = df.drop('Low', axis=1)
    df = df.drop('Close', axis=1)
    df = df.drop('Volume', axis=1)
    
    model = model(df)
    model_train(df, model)
    
    
    
    