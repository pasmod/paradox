# -*- coding: utf-8 -*-
import numpy as np
# np.random.seed(1337)
from paradox.loggers.result_logger import ResultLogger
from paradox.loaders.keras_loader import load_keras_data_set
from paradox.loaders.keras_loader import DataSetType
from paradox.encoders import models
from paradox.evaluation.metrics import evaluate_keras_predictions
from keras.optimizers import SGD, Adagrad, Adam
import time
from keras.callbacks import ModelCheckpoint, EarlyStopping

result_logger = ResultLogger('results.log')
number_of_classes = 3
language = 'Tamil'
data_type = DataSetType.one_hot_encoding_character


def run_simple_model(number_of_classes=number_of_classes, language=language, data_type=data_type, nb_epoch=200,
                     batch_size=128, ngram_range=(1, 1)):
    data_set = load_keras_data_set(language, number_of_classes, data_set_type=data_type, concat_vectors=True,
                                   ngram_range=ngram_range)
    length_input_layer = len(data_set['vocabulary']) * 2
    batch_size = len(data_set['X_train'])
    model = models.simple_model(length_input_layer=length_input_layer, number_of_classes=number_of_classes)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
    checkpoint = ModelCheckpoint("/var/www/trained_models/temp/simple_encoder.model."
                                 "{epoch:03d}-{val_acc:.4f}.hdf5",
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    early_stop = EarlyStopping(monitor='val_acc', patience=20, verbose=1)
    model.fit(data_set['X_train'], data_set['y_train'], callbacks=[checkpoint, early_stop], verbose=1,
              validation_data=(data_set['X_test'], data_set['y_test_categorical']),
              nb_epoch=nb_epoch, batch_size=batch_size)
    avg_f1 = evaluate_keras_predictions(data_set['y_test'], model.predict(data_set['X_test']))
    print(model.evaluate(data_set['X_test'], data_set['y_test_categorical'], batch_size=batch_size))
    result_logger.log_result(locals())
    return model


def run_deep_dense_model(number_of_classes=number_of_classes, language=language, data_type=data_type, nb_epoch=200,
                         batch_size=128):
    data_set = load_keras_data_set(language, number_of_classes, data_set_type=data_type, concat_vectors=True)
    length_input_layer = len(data_set['vocabulary']) * 2
    batch_size = len(data_set['X_train'])
    model = models.deep_dense_model(length_input_layer=length_input_layer, number_of_classes=number_of_classes)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
    checkpoint = ModelCheckpoint("/var/www/trained_models/temp/simple_encoder.model."
                                 "{epoch:03d}-{val_acc:.4f}.hdf5",
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    model.fit(data_set['X_train'], data_set['y_train'], callbacks=[checkpoint], verbose=1,
              validation_data=(data_set['X_test'], data_set['y_test_categorical']),
              nb_epoch=nb_epoch, batch_size=batch_size)
    avg_f1 = evaluate_keras_predictions(data_set['y_test'], model.predict(data_set['X_test']))
    print(model.evaluate(data_set['X_test'], data_set['y_test_categorical'], batch_size=batch_size))
    result_logger.log_result(locals())
    return model


def run_simple_merge_approach(number_of_classes=number_of_classes, language=language, data_type=data_type, nb_epoch=200,
                              batch_size=128):
    data_set = load_keras_data_set(language, number_of_classes, data_set_type=data_type, concat_vectors=False)
    length_input_layer = len(data_set['vocabulary'])
    batch_size = len(data_set['X_train'])
    model = models.simple_merge_approach(length_input_layer=length_input_layer, number_of_classes=number_of_classes)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
    checkpoint = ModelCheckpoint("/var/www/trained_models/temp/simple_merge_approach."
                                 "{epoch:03d}-{val_acc:.4f}.hdf5",
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    model.fit([data_set['X_train'][0], data_set['X_train'][1]], data_set['y_train'], callbacks=[checkpoint], verbose=1,
              validation_data=([data_set['X_test'][0], data_set['X_test'][1]], data_set['y_test_categorical']),
              nb_epoch=nb_epoch, batch_size=batch_size)
    avg_f1 = evaluate_keras_predictions(data_set['y_test'], model.predict(data_set['X_test']))
    print(model.evaluate([data_set['X_test'][0], data_set['X_test'][1]], data_set['y_test_categorical'],
                         batch_size=batch_size))
    result_logger.log_result(locals())
    return model


def run_lstm_approach(number_of_classes=number_of_classes, language=language, data_type=DataSetType.sequence_word,
                      nb_epoch=200,
                      batch_size=128, ngram_range=(1, 1)):
    data_set = load_keras_data_set(language, number_of_classes, data_set_type=data_type, concat_vectors=True,
                                   ngram_range=ngram_range)
    print('data set loaded')
    length_input_layer = len(data_set['vocabulary']) * 2
    batch_size = len(data_set['X_train'])
    model = models.lstm_approach(vocabulary_size=len(data_set['vocabulary']),
                                 sequence_length=2 * data_set['max_length'],
                                 number_of_classes=number_of_classes)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
    checkpoint = ModelCheckpoint("/var/www/trained_models/temp/simple_encoder.model."
                                 "{epoch:03d}-{val_acc:.4f}.hdf5",
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    early_stop = EarlyStopping(monitor='val_acc', patience=20, verbose=1)
    model.fit(data_set['X_train'], data_set['y_train'], callbacks=[checkpoint, early_stop], verbose=1,
              validation_data=(data_set['X_test'], data_set['y_test_categorical']),
              nb_epoch=nb_epoch, batch_size=batch_size)
    avg_f1 = evaluate_keras_predictions(data_set['y_test'], model.predict(data_set['X_test']))
    print(model.evaluate(data_set['X_test'], data_set['y_test_categorical'], batch_size=batch_size))
    result_logger.log_result(locals())
    return model


def run_lstm_branch_approach(number_of_classes=number_of_classes, language=language,
                             data_type=DataSetType.sequence_word,
                             nb_epoch=200,
                             batch_size=128, ngram_range=(1, 1)):
    data_set = load_keras_data_set(language, number_of_classes, data_set_type=data_type, concat_vectors=False,
                                   ngram_range=ngram_range)
    length_input_layer = len(data_set['vocabulary'])
    # batch_size = len(data_set['X_train'])
    batch_size = 32
    model = models.lstm_branch_approach(vocabulary_size=len(data_set['vocabulary']),
                                        sequence_length=data_set['max_length'],
                                        number_of_classes=number_of_classes)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
    checkpoint = ModelCheckpoint("/var/www/trained_models/temp/simple_encoder.model."
                                 "{epoch:03d}-{val_acc:.4f}.hdf5",
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    early_stop = EarlyStopping(monitor='val_acc', patience=20, verbose=1)
    model.fit([data_set['X_train'][0], data_set['X_train'][1]], data_set['y_train'], callbacks=[checkpoint], verbose=1,
              validation_data=([data_set['X_test'][0], data_set['X_test'][1]], data_set['y_test_categorical']),
              nb_epoch=nb_epoch, batch_size=batch_size)
    avg_f1 = evaluate_keras_predictions(data_set['y_test'], model.predict(data_set['X_test']))
    print(model.evaluate([data_set['X_test'][0], data_set['X_test'][1]], data_set['y_test_categorical'],
                         batch_size=batch_size))
    result_logger.log_result(locals())
    return model


# model = run_simple_model()
# model = run_lstm_branch_approach(nb_epoch=10)
# model = run_simple_merge_approach(nb_epoch=10)
# model.save('/var/www/trained_models/temp/{}.model'.format(time.strftime("%Y%m%d_%H%M%S")))

model = run_lstm_branch_approach(language='Malayalam', number_of_classes=2, nb_epoch=20)
model = run_lstm_branch_approach(language='Malayalam', number_of_classes=3, nb_epoch=20)
model = run_lstm_branch_approach(language='Punjabi', number_of_classes=2, nb_epoch=20)
model = run_lstm_branch_approach(language='Punjabi', number_of_classes=3, nb_epoch=20)
model = run_lstm_branch_approach(language='Hindi', number_of_classes=2, nb_epoch=20)
model = run_lstm_branch_approach(language='Hindi', number_of_classes=3, nb_epoch=20)
model = run_lstm_branch_approach(language='Tamil', number_of_classes=2, nb_epoch=20)
model = run_lstm_branch_approach(language='Tamil', number_of_classes=3, nb_epoch=20)
# model = run_deep_dense_model()
