
from keras.models import load_model
import tensorflow as tf
from flask import Flask, render_template, request
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt


def load_lstm_model():
    """Load in the pre-trained model"""
    global model
    # model = load_model('../trained_models/broc_model_trained.h5')
    # model = load_model('../trained_models/carr_model_trained.h5')
    # model = load_model('../trained_models/onn_model_trained.h5')
    # model = load_model('../trained_models/pepp_model_trained.h5')
    # model = load_model('../trained_models/pot_model_trained.h5')
    # model = load_model('../trained_models/app_model_trained.h5')
    # model = load_model('../trained_models/grp_model_trained.h5')
    # model = load_model('../trained_models/orn_model_trained.h5')
    # model = load_model('../trained_models/pr_model_trained.h5')
    # model = load_model('../trained_models/str_model_trained.h5')
    broc_model = load_model('trained_models/broc_model_trained.h5')
    carr_model = load_model('trained_models/carr_model_trained.h5')
    onn_model = load_model('trained_models/onn_model_trained.h5')
    pepp_model = load_model('trained_models/pepp_model_trained.h5')
    pot_model = load_model('trained_models/pot_model_trained.h5')
    app_model = load_model('trained_models/app_model_trained.h5')
    grp_model = load_model('trained_models/grp_model_trained.h5')
    orn_model = load_model('trained_models/orn_model_trained.h5')
    pr_model = load_model('trained_models/pr_model_trained.h5')
    str_model = load_model('trained_models/str_model_trained.h5')

    model.summary
    # Required for model to work
    global graph
    graph = tf.get_default_graph()



if __name__ == "__main__":
    print(("* Loading LSTM model and Flask starting server..."
           "please wait until server has fully started"))
    load_lstm_model()
    # Run app
    app.run(host="0.0.0.0", port=80)
