import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models import word2vec
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import re
import string as st
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')


def remove_stopwords(raw_text):
    stops = set(stopwords.words("english"))
    text = word_tokenize(raw_text)
    text = [w for w in text if not w in stops and len(w) >= 3]
    return text


def load_data(fpath=Path(__file__).resolve().parent / 'data' / 'ExtractedTweets.csv', num_rows=0):
    """
    Loads, cleans and returns the twitter dataset
    :param fpath: path to the csv file
    :param num_rows: number of rows to be randomly extracted, defines the size of the returned dataset
    :return: cleaned dataset of size num_rows
    """

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
    """
    Performs one hot encoding on the dataset
    :param df: dataset of tweets to pass to the function
    :return: Dataset in one hot form
    """
    return df["Tweet"].str.get_dummies(' ')


def w2v(df, method='tfidf'):
    """
    performs word2vec on the dataset
    :param df: dataframe of tweets
    :param method: either avg of tfidf
    :return: new dataset with w2v applied
    """
    tweets = df["Tweet"].copy()
    sentences = [sent.split(' ') for sent in tweets.tolist()]
    model = word2vec.Word2Vec(sentences, min_count=1)
    if method == 'avg':
        out = [np.average([model.wv[word] for word in entry.split(' ')], axis=0) for entry in tweets]
    elif method == 'tfidf':
        corpus = [Dictionary(sentences).doc2bow(tweet) for tweet in sentences]
        tf_idf = TfidfModel(corpus, dictionary=Dictionary(sentences))
        out = [np.average([idf * model.wv[tf_idf.id2word[i]] for i, idf in tf_idf[corpus[tweet_id]]], axis=0)
               for tweet_id in range(len(tweets))]
    else:
        raise RuntimeError(f"Method '{method}' not supported")
    return np.array(out)


def GloVe(df):
    embedding_index = dict()
    fpath = Path(__file__).resolve().parent / 'data' / 'glove.twitter.27B.25d.txt'
    f = open(fpath.as_posix())
    for line in f:
        values = line.split()
        word = values[0]
        coef = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coef
    f.close()

    vocabulary_size = 50000
    tokenizer = Tokenizer(num_words=vocabulary_size)

    df["Tweet"] = df["Tweet"].map(lambda x: remove_stopwords(x))
    df = df.sample(frac=1).reset_index(drop=True)  # shuffles dataset
    tokenizer.fit_on_texts(df["Tweet"])
    sequences = tokenizer.texts_to_sequences(df["Tweet"])
    data = pad_sequences(sequences, maxlen=140)

    embedding_matrix = np.zeros((vocabulary_size, 25))
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    return pd.DataFrame(data, df["Party"])
