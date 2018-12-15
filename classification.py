from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
from config import cv_folds, KNN_gridsearch_params, SVM_gridsearch_params, verbosity

from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
import nltk

import numpy as np


def KNN(x, y):
    """
    Performs KNN classification on the data
    :param x: nxk array of processed tweet data (n tweets, k features)
    :param y: list of n party labels (dem or rep)
    :return:
    """
    model = KNeighborsClassifier()
    gridsearch = GridSearchCV(model, KNN_gridsearch_params, cv=cv_folds, verbose=verbosity)
    gridsearch.fit(x, y)
    return gridsearch


def SVM(x, y):
    """
    Performs SVM classification on the data
    :param x: nxk array of processed tweet data (n tweets, k features)
    :param y: list of n party labels (dem or rep))
    :return:
    """
    model = SVC()
    gridsearch = GridSearchCV(model, SVM_gridsearch_params, cv=cv_folds, verbose=verbosity)
    gridsearch.fit(x, y)
    return gridsearch


def LSTM_model(x, y):

    def labeler(party):
        if party == "Democrat":
            return 1
        else:
            return 0

    labels = y.map(lambda x: labeler(x))
    model = Sequential()
    model.add(Embedding(x.shape[0], 25, input_length=x.shape[1], trainable=True))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='sigmoid'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit train data
    model.fit(x, np.array(labels), validation_split=0.1, epochs=10)
    return model