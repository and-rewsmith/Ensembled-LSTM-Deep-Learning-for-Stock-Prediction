# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from extra_variables import addVariables, add_RSI
from backtest import backtest
from lstm import flatten_dataset, lstm_train
import subprocess
import webbrowser
import time
import os

folder = './Graph'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

proc = subprocess.Popen('tensorboard --logdir=./Graph', shell=True)
time.sleep(5)
#Allow process to run...
webbrowser.open('http://DESKTOP-GCNBT83:6006', new=2)






#Turning off Pandas iloc warning
pd.options.mode.chained_assignment = None  # default='warn'


def get_data(ticker):

    key_list = []
    with open('keys.txt') as keys:
        lines = keys.readlines()
        for line in lines:
            key_list.append(line)
    av_key = key_list[0]
    ts = TimeSeries(av_key, retries=100)
    data, metadata = ts.get_daily_adjusted(ticker, outputsize='full')
    data = pd.DataFrame.from_dict(data)
    print(data)
    df = data.T
    df.columns = ['Open','High','Low','Drop','Close', 'Volume', 'Drop1', 'Drop2']
    df = df.drop(['Drop', 'Drop1', 'Drop2'], axis = 1)
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

    for i in range(len(df)-30):
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

    Open_5d_change = []
    High_5d_change = []
    Low_5d_change = []
    Close_5d_change = []
    Volume_5d_change = []

    for i in range(len(df)-30):
        day = df.iloc[i,:]
        next_day = df.iloc[i+5,:]

        for col in columns:
            data = (next_day[col] - day[col])/day[col]
            if col == 'Open':
                Open_5d_change.append(data)
            if col == 'High':
                High_5d_change.append(data)
            if col == 'Low':
                Low_5d_change.append(data)
            if col == 'Close':
                Close_5d_change.append(data)
            if col == 'Volume':
                Volume_5d_change.append(data)


    df = df.iloc[0:-30,:]

    df['Open_1d_change'] = Open_1d_change
    df['High_1d_change'] = High_1d_change
    df['Low_1d_change'] = Low_1d_change
    df['Close_1d_change'] = Close_1d_change
    df['Volume_1d_change'] = Volume_1d_change

    df['Open_5d_change'] = Open_5d_change
    df['High_5d_change'] = High_5d_change
    df['Low_5d_change'] = Low_5d_change
    df['Close_5d_change'] = Close_5d_change
    df['Volume_5d_change'] = Volume_5d_change


    return df

def moving_average(group):
    sma = group.rolling(9).mean()
    return sma

def thirty_moving_average(group):
    sma = group.rolling(30).mean()
    return sma

def prepare_data(ticker):
    
    df = get_data(ticker)
    df = addVariables(df, ticker)
    df = add_RSI(df)

    ma = moving_average(df.iloc[::-1]['Close'])
    Close_ma = ma.iloc[::-1]
    ma = moving_average(df.iloc[::-1]['Open'])
    Open_ma = ma.iloc[::-1]
    ma = moving_average(df.iloc[::-1]['High'])
    High_ma = ma.iloc[::-1]
    ma = moving_average(df.iloc[::-1]['Low'])
    Low_ma = ma.iloc[::-1]
    ma = moving_average(df.iloc[::-1]['Volume'])
    Volume_ma = ma.iloc[::-1]

    thirty_ma = thirty_moving_average(df.iloc[::-1]['Close'])
    Close_thirty_ma = thirty_ma.iloc[::-1]
    thirty_ma = thirty_moving_average(df.iloc[::-1]['Open'])
    Open_thirty_ma = thirty_ma.iloc[::-1]
    thirty_ma = thirty_moving_average(df.iloc[::-1]['High'])
    High_thirty_ma = thirty_ma.iloc[::-1]
    thirty_ma = thirty_moving_average(df.iloc[::-1]['Low'])
    Low_thirty_ma = thirty_ma.iloc[::-1]
    thirty_ma = thirty_moving_average(df.iloc[::-1]['Volume'])
    Volume_thirty_ma = thirty_ma.iloc[::-1]
    
    ma = moving_average(df.iloc[::-1]['convergence'])
    convergence_ma = ma.iloc[::-1]
    ma = moving_average(df.iloc[::-1]['RSI'])
    RSI_ma = ma.iloc[::-1]
    ma = moving_average(df.iloc[::-1]['MACD_change'])
    MACD_change_ma = ma.iloc[::-1]
    ma = moving_average(df.iloc[::-1]['signal_line'])
    signal_line_ma = ma.iloc[::-1]

    


    df = dataVariables(df, ticker)
    Close_list = df['Close']

    df = df.drop('Open', axis=1)
    df = df.drop('High', axis=1)
    df = df.drop('Low', axis=1)
    df = df.drop('Close', axis=1)
    df = df.drop('Volume', axis=1)
    
    df['Close_ma'] = Close_ma[0:-30]
    df['Open_ma'] = Open_ma[0:-30]
    df['High_ma'] = High_ma[0:-30]
    df['Low_ma'] = Low_ma[0:-30]
    df['Volume_ma'] = Volume_ma[0:-30]
    
    df['Close_thirty_ma'] = Close_thirty_ma[0:-30]
    df['Open_thirty_ma'] = Open_thirty_ma[0:-30]
    df['High_thirty_ma'] = High_thirty_ma[0:-30]
    df['Low_thirty_ma'] = Low_thirty_ma[0:-30]
    df['Volume_thirty_ma'] = Volume_thirty_ma[0:-30]
    
    df['convergence_ma'] = convergence_ma[0:-30]
    df['RSI_ma'] = RSI_ma[0:-30]
    df['MACD_change_ma'] = MACD_change_ma[0:-30]
    df['signal_line_ma'] = signal_line_ma[0:-30]
    
    df.to_csv('data.csv')
    
    return df, Close_list

def create_model(optimizer='sgd', learn_rate=.001, init_mode='uniform',
                 activation1='relu', activation2='relu', activation3='tanh', activation4='sigmoid'):

    model = Sequential()
    model.add(Dense(100, input_dim=32, activation=activation1, kernel_initializer=init_mode))
    model.add(Dropout(0.05))
    model.add(Dense(150, activation=activation2, kernel_initializer=init_mode))
    model.add(Dropout(0.05))
    model.add(Dense(50, activation=activation3, kernel_initializer=init_mode))
    model.add(Dropout(0.05))
    model.add(Dense(10, activation=activation4, kernel_initializer=init_mode))
    model.add(Dropout(0.025))
    model.add(Dense(1))
    
    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    return model


    
def preprocess(df):

    df_zeros = df[df['Next_Change'] == 0]
    df_ones = df[df['Next_Change'] == 1]
    min_length = min([len(df_zeros), len(df_ones)])
    df_zeros = df_zeros.iloc[0:min_length, :]
    df_ones = df_ones.iloc[0:min_length, :]
    df = pd.concat([df_zeros,df_ones])
    Y = df['Next_Change']
    X1 = df.drop('Next_Change', axis=1)
    X = X1
    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

    return X_train, X_test, y_train, y_test, X, Y

def model_train(df, model):

    X_train, X_test, y_train, y_test, X, Y = preprocess(df)
    
    history = model.fit(X_train, y_train, epochs=5, verbose=1, validation_data=(X_test,y_test))
    predictions = model.predict(X)

    plt.figure()
    plt.hist(predictions, bins = 100, normed = True, alpha = 0.5)
    plt.show

    return history, model, df



def plot_model(history):

    plt.figure()
    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':

    ticker = 'PYBI'
    cv_split = 0.05
    
    df, Close_list = prepare_data(ticker)
    
    X_full2d, X_full3d, Y, X_train2d, X_test2d, X_train3d, X_test3d, y_train, y_test, split_value  = flatten_dataset(df, test_size=cv_split)
    
    model, history, predictions, targets = lstm_train(X_full2d, X_full3d, Y, X_train2d, X_test2d, X_train3d, X_test3d, y_train, y_test, batch_size=100, epochs=50)
    
    backtest(predictions, 10000, Close_list, targets, ticker, split_value, sell_tolerance=0.5, buy_tolerance=0.5)




