from data_processing import *
from classification import *

df = load_data(num_rows=30)
one_hot = one_hot(df)
w2v = w2v(df, method='tfidf')

pass