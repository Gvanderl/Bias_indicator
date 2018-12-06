num_tweets = 100
cv_folds = 5
verbosity = 0
KNN_gridsearch_params = {'n_neighbors': [1, 5, 10, 50],
                         'weights': ['uniform', 'distance']}
SVM_gridsearch_params = {'C': [10 ** (1 - i) for i in range(4)],
                         'kernel': ['linear', 'rbf', 'sigmoid', 'poly']}
