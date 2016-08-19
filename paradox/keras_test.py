# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from evaluation.metrics import calculate_and_print_metrics
from keras.optimizers import SGD
import numpy as np
from loaders.keras_loader import load_keras_data_set


number_of_classes = 2
data_set = load_keras_data_set('Punjabi', number_of_classes)
length_input_layer = len(data_set['vocabulary']) * 2


# ***** Keras parameters *****
np.random.seed(123456)
batch_size = 128
nb_epoch = 20


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
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='sigmoid'))
model.add(Dense(number_of_classes, activation='softmax'))
# model.add(Activation('tanh'))
# model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(data_set['X_train'], data_set['y_train'],
          nb_epoch=nb_epoch,
          batch_size=batch_size)

predicted_values = model.predict(data_set['X_test'])
predicted_classes = np.argmax(predicted_values, axis=1)

calculate_and_print_metrics(data_set['y_test'], predicted_classes)
score = model.evaluate(data_set['X_test'], data_set['y_test_categorical'], batch_size=batch_size)

print score
