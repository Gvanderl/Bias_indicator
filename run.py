from data_processing import *
from classification import *

# gets tweet data as dataframe
df = load_data(num_rows=100)
# gets the list of labels, y
sample_labels = df.get('Party')

# we represent the words in 3 different ways: one-hot, word2vec, and glove
one_hot_data = one_hot(df)
w2v_data = w2v(df, method='tfidf')
#glove_data = glove(df)
datas = [one_hot_data, w2v_data, glove_data]

# for each of the 3 representations of the data, we perform classification
# using 4 different methods: KNN, SVM, LSVM, and RNN
for data in datas:
    KNN(data, sample_labels, 40)
    SVM(data, sample_labels)
    #RNN(data, sample_labels)
    #LSVM(data, sample_labels)

pass