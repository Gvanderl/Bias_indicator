from data_processing import *
from classification import *
from config import num_tweets
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


def plot_gridsearch(model, name='model', metric='mean_test_score', save=True):
    params = []
    keys = []
    param_dict = model.param_grid
    for key in param_dict:
        params.append(param_dict[key])
        keys.append(key)
    scores = model.cv_results_[metric].reshape(len(params[0]), len(params[1]))
    df = pd.DataFrame(scores, index=params[0], columns=params[1])
    ax = sns.heatmap(df, annot=True, cmap="OrRd", cbar=False)
    # ax.collections[0].colorbar.set_label(metric)
    ax.set(xlabel = keys[1], ylabel=keys[0])
    plt.title('Gridsearch results for {}'.format(name))
    if save:
        plt.savefig('output/Gridsearch_{}'.format(name), bbox_inches='tight', dpi=200)
    plt.show()

# Gets tweet data as dataframe
df = load_data(num_rows=num_tweets)
# Gets the labels, y
sample_labels = df.get('Party')

embeddings = \
    {
        "One hot": one_hot(df)
        , "Word2Vec average": w2v(df, method='avg')
        , "Word2Vec tfidf": w2v(df, method='tfidf')
        # , "GloVe": GloVe(df)
    }
classifiers = \
    {
        "KNN": KNN
        , "SVM": SVM
    }

print("\n********** Running grid search on all models to find best parameters **********\n")
models = dict()

# For sklearn classifiers
for classifier in classifiers.keys():
    print(f"\nTesting classifer '{classifier}'")
    for embedding in embeddings.keys():
        print(f"Testing embedding '{embedding}'")
        # models[embedding + ' with ' + classifier] = classifiers[classifier](embeddings[embedding], sample_labels)

print("\n********** Grid search finished! **********\n")

print("Results:\n")

for model_name, model in models.items():
    print(f"Accuracy for {model_name}: {model.best_score_*100}%")
    plot_gridsearch(model, name=model_name)

print("Testing LSTM:")

# For LSTM
for embedding in embeddings.keys():
    print(f"Testing embedding '{embedding}' with LSTM")
    LSTM = LSTM_model(embeddings[embedding], sample_labels)



