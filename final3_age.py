import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
import datetime, time, json
from string import punctuation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda, Activation, LSTM, Flatten, Convolution1D, GRU, MaxPooling1D
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import initializers
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adadelta, Adam
from collections import defaultdict
from keras.utils import np_utils
from keras.layers.advanced_activations import PReLU

import codecs
import random

corpus = pd.read_csv("features_combined.csv")
#random.shuffle(corpus,lambda: 0.374)
np.random.seed(44)
corpus = corpus.reindex(np.random.permutation(corpus.index))
#corpus = corpus.head(10000)

corpus.describe()
print (corpus.shape)


#f0 = corpus.WC




#print f0[:10]
#print("**************")
#print f1[0]
#print f1[2]

#>>> for a in xrange(93):
#...     print "f"+str(a)+"= corpus.iloc[:"+str(a+5)+"]"


#f_tr = []

f_tr = corpus.as_matrix(columns=corpus.columns[5:])
f_normed = f_tr / f_tr.max(axis=0)
f_tr = f_normed
#f_tr = np.array(f_tr)

print ("f_tr shapes")
print (f_tr.shape)
#print (f_tr[0][0])
#print (f_tr[0][1])
#print (f_tr[0][-1])

#print (a)
'''
print("Corpus shape ", corpus.shape)
def process_blog(post_list, posts, post_list_name, dataframe):
    ''transform blogs and display progress''
    print("For ", post_list_name)
    print("num of Qs = ", len(posts))
    print("dataframe len =", len(dataframe))
    for post in posts:
        post_list.append(str(post))
        if len(post_list) % 100000 == 0:
            print("Q list len = ", len(post_list))
            print("Dataframe len = ", len(dataframe))

corpus_post = []
process_blog(corpus_post, corpus.post, 'corpus_post', corpus)

print (corpus_post[0][:100])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus_post)
corpus_post_word_sequences = tokenizer.texts_to_sequences(corpus_post)
word_index = tokenizer.word_index
print('*********************************************')
print("Words in index: %d" % len(word_index))
print (corpus_post[0][:100])


#>>> for a in xrange(93):
#...     print "f"+str(a)+"= corpus.iloc[:"+str(a+5)+"]"

# Pad the posts so that they all have the same length.

max_post_len = 4000
print (len(corpus_post[0]))
corpus_post = pad_sequences(corpus_post_word_sequences, 
                              maxlen = max_post_len)
print("corpus_post is complete.")
print (corpus_post[0][:30])

embeddings_index = {}
with codecs.open('glove.840B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))

embedding_dim = 300

nb_words = len(word_index)
word_embedding_matrix = np.zeros((nb_words + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        word_embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

'''

y_label = [2 if g >30 else (1 if g>20 else 0) for g in corpus.age]
#y_label = [1 if g == 'male' else 0 for g in corpus.gender]
y_train = np_utils.to_categorical(y_label)
#y_train = np.array(y_label)
print (y_train.shape)
#print (corpus_post.shape)
units = 256 # Number of nodes in the Dense layers
dropout = 0.25 # Percentage of nodes to drop
nb_filter = 64 # Number of filters to use in Convolution1D
filter_length = 3 # Length of filter for Convolution1D
# Initialize weights and biases for the Dense layers
weights = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=2)
bias = bias_initializer='zeros'
print('randominzing')
'''
c = list(zip(corpus_post, y_train))
random.shuffle(c,lambda: 0.374)

corpus_post, y_train = zip(*c)
c=[]
print (y_train.shape)
print (corpus_post.shape)
'''
print ('starting model')

#act = PReLU(init='zero', weights=None)
'''
model1 = Sequential()
model1.add(Embedding(nb_words + 1,
                     embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = max_post_len,
                     trainable = False))

model1.add(Convolution1D(filters = nb_filter, 
                         kernel_size = filter_length, 
                         padding = 'same'))
model1.add(BatchNormalization())
#model1.add(act)
model1.add(Activation('relu'))
model1.add(Dropout(dropout))

model1.add(Convolution1D(filters = nb_filter, 
                         kernel_size = filter_length, 
                         padding = 'same'))
model1.add(BatchNormalization())
#model1.add(act)
model1.add(Activation('relu'))
model1.add(Dropout(dropout))

model1.add(Flatten())
''
model2 = Sequential()
model2.add(Embedding(nb_words + 1,
                     embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = max_post_len,
                     trainable = False))
#model6.add(Embedding(nb_words + 1, embedding_dim, input_length=max_question_lenth, dropout=0.2))
#model.add(LSTM(embedding_dim,return_sequences=True))
#model.add(BatchNormalization())
#model6.add(Activation('relu'))
#model.add(Dropout(dropout))
model2.add(LSTM(embedding_dim))
#model2.add(BatchNormalization())
#model6.add(Activation('relu'))
model2.add(Dropout(dropout))
''
model3 = Sequential()
model3.add(Embedding(nb_words + 1,
                     embedding_dim,
                     weights = [word_embedding_matrix],
                     input_length = max_post_len,
                     trainable = False))
model3.add(TimeDistributed(Dense(embedding_dim)))
model3.add(BatchNormalization())
model3.add(Activation('relu'))
model3.add(Dropout(dropout))
model3.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim, )))
'''
model = Sequential()
model.add(Dense(units*4, kernel_initializer=weights, bias_initializer=bias, input_shape=(1093,), name="my_dense_1"))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(BatchNormalization())
model.add(Dense(units*2, kernel_initializer=weights, bias_initializer=bias, name="my_dense_2"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))
'''
model = Sequential()
#model.add(Merge([model1, model2, model3, model4], mode='concat'))
model.add(Merge([model1, model3, model4], mode='concat'))
model.add(Dense(units*2, kernel_initializer=weights, bias_initializer=bias))
model.add(BatchNormalization())
#model.add(act)
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(units, kernel_initializer=weights, bias_initializer=bias))
model.add(BatchNormalization())
#model.add(act)
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(units, kernel_initializer=weights, bias_initializer=bias))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))
'''
model.add(Dense(units, kernel_initializer=weights, bias_initializer=bias, name="my_dense_3"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(units, kernel_initializer=weights, bias_initializer=bias, name="my_dense_4"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(3, kernel_initializer=weights, bias_initializer=bias, name="my_dense_5"))
model.add(BatchNormalization())
model.add(Activation('softmax'))

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#opt = Adadelta()
#opt = SGD(lr=0.01)
opt = Adam(lr = 0.001)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics=['accuracy'])
model.summary()


t0 = time.time()
save_best_weights = 'final3_age.h5'

#test_corpus_post = corpus_post[int(len(corpus_post) * 0.9):]
test_y = y_train[int(len(y_train) * 0.9):]
#corpus_post = corpus_post[:int(len(corpus_post) * 0.9)]
y_train = y_train[:int(len(y_train) * 0.9)]
f_te = f_tr[int(len(f_tr) * 0.9):]
f_tr = f_tr[:int(len(f_tr) * 0.9)]
print ('testing shapes')
#print (test_corpus_post.shape)
print (test_y.shape)

print('feature shapes')
print (f_tr.shape)
print (f_te.shape)
#print (f_tr[0].shape)



callbacks = [ModelCheckpoint(save_best_weights, monitor='val_acc', save_best_only=True),
             EarlyStopping(monitor='val_acc', patience=50, verbose=1, mode='auto')]
#history = model.fit([corpus_post, corpus_post, corpus_post, f_tr],
#history = model.fit([corpus_post, corpus_post, f_tr],
history = model.fit(f_tr,
                    y_train,
                    batch_size=512,
                    epochs=1000, #Use 100, I reduce it for Kaggle,
                    validation_split=0.15,
                    verbose=True,
                    shuffle=True,
                    callbacks=callbacks)
t1 = time.time()
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
model.load_weights(save_best_weights)
score = model.evaluate(f_te, test_y ,batch_size=1024, verbose=0)
#score = model.evaluate([test_corpus_post, test_corpus_post, f_te], test_y ,batch_size=8, verbose=0)
#score = model.evaluate([test_corpus_post, test_corpus_post, test_corpus_post, f_te], test_y ,batch_size=512, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

