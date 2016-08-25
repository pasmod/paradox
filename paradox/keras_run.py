# -*- coding: utf-8 -*-
import numpy as np
# np.random.seed(1337)
from loaders.keras_loader import load_keras_data_set
from loaders.keras_loader import DataSetType
from encoders import models
from optimizer.optimizer_selector import compile_optimizer
from evaluation.metrics import evaluate_keras_predictions

import time

number_of_classes = 2
language = 'Punjabi'
data_type = DataSetType.one_hot_encoding_character


def run_simple_model(number_of_classes=number_of_classes, language=language, data_type=data_type, nb_epoch=200,
                     batch_size=128):
    data_set = load_keras_data_set(language, number_of_classes, data_set_type=data_type, concat_vectors=True)
    length_input_layer = len(data_set['vocabulary']) * 2
    model = models.simple_model(length_input_layer=length_input_layer, number_of_classes=number_of_classes)
    compile_optimizer('adagrad', model)
    model.fit(data_set['X_train'], data_set['y_train'],
              nb_epoch=nb_epoch,
              batch_size=batch_size)
    evaluate_keras_predictions(data_set['y_test'], model.predict(data_set['X_test']))
    print model.evaluate(data_set['X_test'], data_set['y_test_categorical'], batch_size=batch_size)
    return model


model = run_simple_model()
model.save('/var/www/trained_models/temp/{}.model'.format(time.strftime("%Y%m%d_%H%M%S")))
