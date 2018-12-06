import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.svm import SVC  


# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
def KNN(x, y, num_ks=10, test_size=0.20, n_neighbors=5):
    """
    Performs KNN classification on the data
    :param x: nxk array of processed tweet data (n tweets, k features)
    :param y: list of n party labels (dem or rep)
    :param num_ks: number of k-values to test (see which gets best error rate)
    :param test_size: for corss validation
    :param n_neighbors: for KNN
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)  
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)  
    classifier.fit(x_train, y_train)  
    y_pred = classifier.predict(x_test)  
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred))
    
    error = []
    # Calculating error for K values between 1 and num_ks
    for i in range(1, num_ks):  
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        error.append(np.mean(pred_i != y_test))
        
    plt.figure(figsize=(12, 6))  
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')  
    plt.xlabel('K Value')  
    plt.ylabel('Mean Error') 


def LSVM():
    pass


def SVM(x, y, test_size=0.20):
    """
    Performs SVM classification on the data
    :param x: nxk array of processed tweet data (n tweets, k features)
    :param y: list of n party labels (dem or rep))
    :param test_size: For crossvalidation
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(x_train, y_train) 
    y_pred = svclassifier.predict(x_test)  
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))  


def RNN():
    pass
