import pandas as pd
from pathlib import Path
from gensim.models import word2vec


def load_data(fpath = Path(__file__).resolve().parent / 'data' / 'ExtractedTweets.csv', num_rows=0):
    df = pd.read_csv(fpath.resolve())
    if num_rows > 0:
        df = df.sample(n=num_rows)
    return df


def one_hot(df):
    return df.get_dummies(' ')


def w2v():
    pass


def GloVe():
    pass
