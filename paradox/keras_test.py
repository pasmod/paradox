# -*- coding: utf-8 -*-
from loaders.corpus_loader import load_all_languages
from evaluation.training_test_split import split_training_data
from utils.one_hot_encoder import get_one_hot_encoding
from utils.vocabulary_extractor import get_vocabulary
from keras.models import Graph
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from evaluation.metrics import calculate_and_print_metrics
from keras.optimizers import SGD
import codecs
import sys
import numpy as np
from keras.utils.np_utils import to_categorical

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

data_sets = load_all_languages()
language = 'Punjabi'
task = 'Task2'
test_train_split = split_training_data(data_sets[language][task][0], data_sets[language][task][1])

vocabulary = get_vocabulary(test_train_split['X_train'], analyzer='char_wb')

# transform sentences to character one-hot-encodings

number_of_classes = 3

X_train = get_one_hot_encoding(test_train_split['X_train'], vocabulary=vocabulary, analyzer='char_wb')
y_train = test_train_split['y_train']
#print len(X_train)
X_test = get_one_hot_encoding(test_train_split['X_test'], vocabulary=vocabulary, analyzer='char_wb')
y_test = test_train_split['y_test']
# print len(y_test)


y_train = np.array(y_train)
y_train = to_categorical(y_train, number_of_classes)
y_test_categorical = to_categorical(np.array(y_test), number_of_classes)

# print y_train.shape
#np.random.seed(123456)

batch_size = 128
nb_epoch = 20
length_input_layer = len(vocabulary) * 2


# ****************************
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, input_dim=length_input_layer, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, init='uniform'))
#model.add(Dropout(0.5))
#model.add(Dense(2, activation='sigmoid'))
model.add(Dense(number_of_classes, activation='softmax'))
# model.add(Activation('tanh'))
#model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train,
          nb_epoch=nb_epoch,
          batch_size=batch_size)

predicted_values = model.predict(X_test)
predicted_classes = np.argmax(predicted_values, axis=1)


calculate_and_print_metrics(y_test, predicted_classes)
score = model.evaluate(X_test, y_test_categorical, batch_size=batch_size)

print score
