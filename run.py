from data_processing import *
from classification import *
from config import num_tweets
import warnings
warnings.filterwarnings("ignore")

# Gets tweet data as dataframe
df = load_data(num_rows=num_tweets)
# Gets the labels, y
sample_labels = df.get('Party')

embeddings = \
    {
        "One hot": one_hot(df)
        , "Word2Vec tfidf": w2v(df, method='tfidf')
        , "Word2Vec average": w2v(df, method='avg')
        # , "GloVe": glove(df)
    }
classifiers = \
    {
        "KNN": KNN
        , "SVM": SVM
    }

print("\n********** Running grid search on all models to find best parameters **********\n")
models = dict()
for classifier in classifiers.keys():
    print(f"\nTesting classifer '{classifier}'")
    for embedding in embeddings.keys():
        print(f"Testing embedding '{embedding}'")
        models[embedding + ' with ' + classifier] = classifiers[classifier](embeddings[embedding], sample_labels)
print("\n********** Grid search finished! **********\n")

print("Results:\n")
for model_name, model in models.items():
    print(f"Accuracy for {model_name}: {model.best_score_*100}%")
    # TODO display nice plots using model.cv_results_

