import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import seaborn as sns
import tensorflow.keras
from keras.models import Sequential
from keras.optimizers import nadam 
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM, Activation
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend
import datetime as dt
from dateutil.relativedelta import relativedelta
# import sqlalchemy
# from sqlalchemy.ext.automap import automap_base
# from sqlalchemy.orm import Session
# from sqlalchemy import create_engine, func, inspect
# from sqlalchemy import desc


def broc_load_data(filename, seq_len, normalise_window):
    broc_data = pd.read_csv("dset/broc.csv")
    value = broc_data['value'].values
    seq_len = 10
    sequence_length = seq_len + 1

    result = []
    for index in range(len(value) - sequence_length):
        result.append(value[index: index + sequence_length])
        sequence_length = seq_len + 1
        

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)


    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]

    x_train.shape, x_test.shape

    return [x_train, y_train, x_test, y_test]

def carr_load_data(filename, seq_len, normalise_window):
    carr_data = pd.read_csv("dset/carr.csv")
    value = carr_data['value'].values
    seq_len = 10
    sequence_length = seq_len + 1

    result = []
    for index in range(len(value) - sequence_length):
        result.append(value[index: index + sequence_length])
        sequence_length = seq_len + 1
        

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)


    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]

    x_train.shape, x_test.shape

    return [x_train, y_train, x_test, y_test]

def onn_load_data(filename, seq_len, normalise_window):
    onn_data = pd.read_csv("dset/onn.csv")
    value = onn_data['value'].values
    seq_len = 10
    sequence_length = seq_len + 1

    result = []
    for index in range(len(value) - sequence_length):
        result.append(value[index: index + sequence_length])
        sequence_length = seq_len + 1
        

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)


    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]

    x_train.shape, x_test.shape

    return [x_train, y_train, x_test, y_test]

def pepp_load_data(filename, seq_len, normalise_window):
    pepp_data = pd.read_csv("dset/pepp.csv")
    value = pepp_data['value'].values
    seq_len = 10
    sequence_length = seq_len + 1

    result = []
    for index in range(len(value) - sequence_length):
        result.append(value[index: index + sequence_length])
        sequence_length = seq_len + 1
        

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)


    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]

    x_train.shape, x_test.shape

    return [x_train, y_train, x_test, y_test]

def pot_load_data(filename, seq_len, normalise_window):
    pot_data = pd.read_csv("dset/pot.csv")
    value = pot_data['value'].values
    seq_len = 10
    sequence_length = seq_len + 1

    result = []
    for index in range(len(value) - sequence_length):
        result.append(value[index: index + sequence_length])
        sequence_length = seq_len + 1
        

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)


    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]

    x_train.shape, x_test.shape

    return [x_train, y_train, x_test, y_test]

def app_load_data(filename, seq_len, normalise_window):
    app_data = pd.read_csv("dset/app.csv")
    value = app_data['value'].values
    seq_len = 10
    sequence_length = seq_len + 1

    result = []
    for index in range(len(value) - sequence_length):
        result.append(value[index: index + sequence_length])
        sequence_length = seq_len + 1
        

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)


    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]

    x_train.shape, x_test.shape

    return [x_train, y_train, x_test, y_test]

def grp_load_data(filename, seq_len, normalise_window):
    grp_data = pd.read_csv("dset/grp.csv")
    value = grp_data['value'].values
    seq_len = 10
    sequence_length = seq_len + 1

    result = []
    for index in range(len(value) - sequence_length):
        result.append(value[index: index + sequence_length])
        sequence_length = seq_len + 1
        

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)


    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]

    x_train.shape, x_test.shape

    return [x_train, y_train, x_test, y_test]

def orn_load_data(filename, seq_len, normalise_window):
    orn_data = pd.read_csv("dset/orn.csv")
    value = orn_data['value'].values
    seq_len = 10
    sequence_length = seq_len + 1

    result = []
    for index in range(len(value) - sequence_length):
        result.append(value[index: index + sequence_length])
        sequence_length = seq_len + 1
        

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)


    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]

    x_train.shape, x_test.shape

    return [x_train, y_train, x_test, y_test]

def pr_load_data(filename, seq_len, normalise_window):
    pr_data = pd.read_csv("dset/pr.csv")
    value = pr_data['value'].values
    seq_len = 10
    sequence_length = seq_len + 1

    result = []
    for index in range(len(value) - sequence_length):
        result.append(value[index: index + sequence_length])
        sequence_length = seq_len + 1
        

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)


    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]

    x_train.shape, x_test.shape

    return [x_train, y_train, x_test, y_test]

def str_load_data(filename, seq_len, normalise_window):
    str_data = pd.read_csv("dset/str.csv")
    value = str_data['value'].values
    seq_len = 10
    sequence_length = seq_len + 1

    result = []
    for index in range(len(value) - sequence_length):
        result.append(value[index: index + sequence_length])
        sequence_length = seq_len + 1
        

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)


    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]

    x_train.shape, x_test.shape

    return [x_train, y_train, x_test, y_test]


def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

# build model

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse",metrics=['mean_squared_error'],optimizer="nadam")
    print ("Compilation Time : ", time.time() - start)
    print (model.summary)
    return model

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    print('yo')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in xrange(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in xrange(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in xrange(len(data)/prediction_len):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in xrange(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs