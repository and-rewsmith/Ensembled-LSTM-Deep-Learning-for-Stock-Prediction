# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 00:41:57 2018

@author: Brandon
"""

from sklearn.preprocessing import StandardScaler
from numpy import array
from keras.models import  Model
from keras.layers import Dense, LSTM, Dropout, Input, concatenate
import matplotlib.pyplot as plt
from keras.utils import plot_model
import pandas as pd
from keras.callbacks import TensorBoard


def flatten_dataset(df, test_size=.1):
    
    
    
    
    
    df = df[::-1]
    samples = len(df)
    
    split_value = round(samples - samples*test_size)
    inputs = len(df.columns) - 1
    
    scaler = StandardScaler()
    
    Y = list(df['Next_Change'])
    
    y_train = array(Y[0:split_value])
    y_test = array(Y[split_value:])
    
    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)
    
    
    
    X1 = df.drop('Next_Change', axis = 1)
    X = X1
    
    
    
    X_train = X.iloc[0:split_value, :].values
    X_test = X.iloc[split_value:, :].values
    
    X_train2d = scaler.fit_transform(X_train)
    X_test2d = scaler.fit_transform(X_test)
    X_full2d = scaler.fit_transform(X)
    
    X_train3d = X_train2d.reshape(len(y_train), inputs, 1)
    X_test3d = X_test2d.reshape(len(y_test), inputs, 1)
    X_full3d = X_full2d.reshape(len(Y), inputs, 1)
    
    return X_full2d, X_full3d, Y, X_train2d, X_test2d, X_train3d, X_test3d, y_train, y_test 
    
def lstm_train(X_full2d, X_full3d, Y, X_train2d, X_test2d, X_train3d, X_test3d, y_train, y_test, batch_size):
 
    
    tbCallBack = TensorBoard(log_dir='./Graph', write_graph=True, histogram_freq=5)
    
    # This returns a tensor
    inputs3d = Input(shape=(32,1,))
    x1 = LSTM(16, activation='sigmoid')(inputs3d)
    x1 = Dropout(0.4)(x1)
#    x1 = Dense(1, activation='tanh')(x1)
    
    x2 = LSTM(32, activation='sigmoid')(inputs3d)
    x2 = Dropout(0.4)(x2)
#    x2 = Dense(1, activation='tanh')(x2)
    
    x3 = LSTM(64, activation='sigmoid')(inputs3d)
    x3 = Dropout(0.4)(x3)
    
#    x3 = Dense(1, activation='tanh')(x3)
    ensemble3d = concatenate([x1, x2, x3])
    
    predictions3d = Dense(100, activation='tanh')(ensemble3d)
    predictions3d = Dropout(0.4)(predictions3d)
    predictions3d = Dense(50, activation='sigmoid')(predictions3d)
    predictions3d = Dropout(0.4)(predictions3d)
    predictions3d = Dense(25, activation='tanh')(predictions3d)
    
    inputs2d = Input(shape=(32,))
    
    # a layer instance is callable on a tensor, and returns a tensor
    x2d1 = Dense(75, activation='tanh')(inputs2d)
    x2d1 = Dropout(0.4)(x2d1)
    x2d1 = Dense(150, activation='tanh')(x2d1)
    x2d1 = Dropout(0.4)(x2d1)
    x2d1 = Dense(25, activation='sigmoid')(x2d1)
    
    # a layer instance is callable on a tensor, and returns a tensor
    x2d2 = Dense(50, activation='tanh', input_dim=32)(inputs2d)
    x2d2 = Dropout(0.4)(x2d2)
    x2d2 = Dense(100, activation='sigmoid')(x2d2)
    x2d2 = Dropout(0.4)(x2d2)
    x2d2 = Dense(50, activation='tanh')(x2d2)
    x2d2 = Dropout(0.4)(x2d2)
    x2d2 = Dense(25, activation='sigmoid')(x2d2)
    
    ensemble2d = concatenate([x2d1, x2d2])
    
    predictions2d = Dense(25, activation='sigmoid')(ensemble2d)
    
    full_ensemble = concatenate([predictions3d, predictions2d])
    
    predictions = Dense(1, activation = 'sigmoid')(full_ensemble)
    
    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[inputs2d,inputs3d], outputs=predictions)
    
    model.compile(optimizer='RMSprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    plot_model(model, to_file='plot.png', show_shapes = True)
    
    history = model.fit([X_train2d, X_train3d], y_train, epochs  = 250, batch_size = batch_size, verbose = 1, validation_data = ([X_test2d, X_test3d], y_test), callbacks=[tbCallBack])  # starts training


    predictions = model.predict([X_full2d, X_full3d])
    
    targets = Y
    
    predictions = list(predictions)
    
    predictions = [value[0] for value in predictions]
    
    
    
    pred_df = pd.DataFrame({'Predictions': predictions,
                             'Targets': targets})
    
    ones = pred_df['Predictions'][pred_df['Targets'] == 1]
    zeros = pred_df['Predictions'][pred_df['Targets'] == 0]
    
    
    plt.figure()
    plt.hist(ones, alpha = 0.5, bins = 50, label = 'Green Day Predictions', color = 'green')
    plt.hist(zeros, alpha = 0.5, bins = 50, label = 'Red Day Predictions', color = 'red')
    plt.legend(loc='upper right')
    plt.title('Distirbution of Predictions by Target')
    plt.show()
 
#    model = Sequential()
#    model.add(LSTM(output_dim=512, input_shape=(32, 1), activation='sigmoid', return_sequences=True, unroll=True))
#    model.add(Dropout(0.0))
#    model.add(LSTM(output_dim=256, activation='tanh'))
#    model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
#    
#    model.compile(optimizer = 'RMSprop', metrics = ['accuracy'], loss = ['mean_squared_error'])
#    
#    history = model.fit(X_train, y_train, epochs  = 25, verbose = 1, validation_data = (X_test, y_test))
#    
#    predictions = model.predict(X_test)

    return model, history
    