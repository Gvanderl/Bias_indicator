import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models import word2vec
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import re
import string as st


def load_data(fpath = Path(__file__).resolve().parent / 'data' / 'ExtractedTweets.csv', num_rows=0):

    def clean(string):
        emojis = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
        punctuation = (st.punctuation.replace('@', '').replace('#', '')) + '"' + "'" + '”' + '“' + '‘'
        trans = str.maketrans('', '', punctuation)
        string = str(string).lower()
        string = string.translate(trans)
        string = string.split()
        to_remove = []
        for word in string:
            if word[0] == '#' or word[0] == '@' or word == 'rt' or word[:4] == 'http' or word[0].isnumeric():
                to_remove.append(word)
        for word in to_remove:
            string.remove(word)
        text = emojis.sub(r'', ' '.join(string))
        text = re.sub("[^a-zA-Z ]", "", text)
        return text

    df = pd.read_csv(fpath.resolve())
    if num_rows > 0:
        df = df.sample(n=num_rows)
    df["Tweet"] = df["Tweet"].apply(clean)
    return df


def one_hot(df):
    return df["Tweet"].str.get_dummies(' ')


def w2v(df, method='tfidf', fweights='w2v_weights'):
    tweets = df["Tweet"].copy()
    sentences = [sent.split(' ') for sent in tweets.tolist()]
    if Path(fweights).exists():
        model = word2vec.Word2Vec.load(fweights)
    else:
        model = word2vec.Word2Vec(sentences, min_count=1)
    if method == 'avg':
        out = [np.average([model.wv[word] for word in entry.split(' ')], axis=0) for entry in tweets]
    elif method == 'tfidf':
        corpus = [Dictionary(sentences).doc2bow(tweet)for tweet in sentences]
        tf_idf = TfidfModel(corpus, dictionary=Dictionary(sentences))
        out = [np.average([idf * model.wv[tf_idf.id2word[i]] for i, idf in tf_idf[corpus[tweet_id]]], axis=0)
               for tweet_id in range(len(tweets))]
    else:
        raise RuntimeError(f"Method '{method}' not supported")
    return np.array(out)


def GloVe():
    pass