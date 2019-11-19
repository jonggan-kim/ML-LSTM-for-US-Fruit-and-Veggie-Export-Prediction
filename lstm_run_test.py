{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<bound method Network.summary of <keras.engine.sequential.Sequential object at 0xb52126f60>>\n"
     ]
    },
    {
     "ename": "NameError",
     "evalue": "name 'x_train' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-8-6804973c9453>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     33\u001b[0m     \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"Normal Neural Network - Loss: {broc_model_loss}, Accuracy: {metrics}\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     34\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 35\u001b[0;31m \u001b[0mload_lstm_model\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m<ipython-input-8-6804973c9453>\u001b[0m in \u001b[0;36mload_lstm_model\u001b[0;34m()\u001b[0m\n\u001b[1;32m     30\u001b[0m     \u001b[0;32mglobal\u001b[0m \u001b[0mgraph\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     31\u001b[0m     \u001b[0mgraph\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_default_graph\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 32\u001b[0;31m     \u001b[0mbroc_model_loss\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmetrics\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mbroc_model\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mevaluate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mverbose\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     33\u001b[0m     \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"Normal Neural Network - Loss: {broc_model_loss}, Accuracy: {metrics}\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     34\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'x_train' is not defined"
     ]
    }
   ],
   "source": [
    "\n",
    "from keras.models import load_model\n",
    "import tensorflow as tf\n",
    "from flask import Flask, render_template, request\n",
    "import time\n",
    "import warnings\n",
    "import numpy as np\n",
    "from numpy import newaxis\n",
    "from keras.layers.core import Dense, Activation, Dropout\n",
    "from keras.layers.recurrent import LSTM\n",
    "from keras.models import Sequential\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def load_lstm_model():\n",
    "    global model\n",
    "    broc_model = load_model('broc_model_trained.h5')\n",
    "    carr_model = load_model('carr_model_trained.h5')\n",
    "    onn_model = load_model('onn_model_trained.h5')\n",
    "    pepp_model = load_model('pepp_model_trained.h5')\n",
    "#     pot_model = load_model('pot_model_trained.h5')\n",
    "    app_model = load_model('app_model_trained.h5')\n",
    "    grp_model = load_model('grp_model_trained.h5')\n",
    "    orn_model = load_model('orn_model_trained.h5')\n",
    "    pr_model = load_model('pr_model_trained.h5')\n",
    "    str_model = load_model('str_model_trained.h5')\n",
    "\n",
    "\n",
    "    print(broc_model.summary)\n",
    "    # Required for model to work\n",
    "    global graph\n",
    "    graph = tf.get_default_graph()\n",
    "    broc_model_loss, metrics = broc_model.evaluate(x_train, y_train, verbose=2)\n",
    "    print(f\"Normal Neural Network - Loss: {broc_model_loss}, Accuracy: {metrics}\")\n",
    "\n",
    "load_lstm_model()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
