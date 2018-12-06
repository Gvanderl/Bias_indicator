from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
from config import cv_folds, KNN_gridsearch_params, SVM_gridsearch_params, verbosity


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


def RNN():
    pass
