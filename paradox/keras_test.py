# -*- coding: utf-8 -*-
from evaluation.metrics import calculate_and_print_metrics
import numpy as np
from loaders.keras_loader import load_keras_data_set
from models.model_selector import get_model
from optimizer.optimizer_selector import compile_optimizer
from evaluation.metrics import evaluate_keras_predictions

number_of_classes = 2
data_set = load_keras_data_set('Punjabi', number_of_classes)
length_input_layer = len(data_set['vocabulary']) * 2

# ***** Keras parameters *****
np.random.seed(123456)
batch_size = 128
nb_epoch = 20
# ****************************
model = get_model('first_approach', length_input_layer=length_input_layer, number_of_classes=number_of_classes)
compile_optimizer('svg', model)

model.fit(data_set['X_train'], data_set['y_train'],
          nb_epoch=nb_epoch,
          batch_size=batch_size)

evaluate_keras_predictions(data_set['y_test'], model.predict(data_set['X_test']))
print model.evaluate(data_set['X_test'], data_set['y_test_categorical'], batch_size=batch_size)
