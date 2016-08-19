# -*- coding: utf-8 -*-
from evaluation.metrics import calculate_and_print_metrics
from keras.optimizers import SGD
import numpy as np
from loaders.keras_loader import load_keras_data_set
from models.model_selector import get_model

number_of_classes = 2
data_set = load_keras_data_set('Punjabi', number_of_classes)
length_input_layer = len(data_set['vocabulary']) * 2

# ***** Keras parameters *****
np.random.seed(123456)
batch_size = 128
nb_epoch = 20
# ****************************
model = get_model('first_approach', length_input_layer=length_input_layer, number_of_classes=number_of_classes)

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
