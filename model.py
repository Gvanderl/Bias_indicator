from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding

import numpy as np
import pandas as pd
import regex as re
import string as st
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import regex as re
import string as st


def load_data(fpath = '/Users/ravinagda/Desktop/fall18/lign167/project/democratvsrepublicantweets/ExtractedTweets.csv', num_rows=0):
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

    df = pd.read_csv(fpath)
    if num_rows > 0:
        df = df.sample(n=num_rows)
    df["Tweet"] = df["Tweet"].apply(clean)
    return df

def remove_stopwords(raw_text):
    stops = set(stopwords.words("english"))
    text = word_tokenize(raw_text)
    text = [w for w in text if not w in stops and len(w) >= 3]
    return text

def labeler(party):
    if party == "Democrat":
        return 1
    else:
        return 0

embedding_index = dict()
f = open('/Users/ravinagda/Desktop/fall18/lign167/project/glove.twitter.27B.25d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coef = np.asarray(values[1:], dtype = 'float32')
    embedding_index[word] = coef
f.close()

vocabulary_size = 50000
tokenizer = Tokenizer(num_words=vocabulary_size)

df = load_data()
df["Tweet"] = df["Tweet"].map(lambda x: remove_stopwords(x))
df = df.sample(frac=1).reset_index(drop=True) #shuffles dataset
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


labels = df["Party"].map(lambda x: labeler(x))

model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, 25, input_length=140, weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='sigmoid'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(100))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## Fit train data
model_glove.fit(data, np.array(labels), validation_split=0.1, epochs = 10)
