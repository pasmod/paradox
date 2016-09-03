# -*- coding: utf-8 -*-
import numpy as np
# np.random.seed(1337)
from paradox.loggers.result_logger import ResultLogger
from paradox.loaders.keras_loader import load_keras_train_data_set, load_keras_test_data_set
from paradox.encoders import models
from paradox.evaluation.metrics import evaluate_keras_predictions
from keras.optimizers import SGD, Adagrad, Adam
import time
from keras.callbacks import ModelCheckpoint, EarlyStopping
import logging

result_logger = ResultLogger('results.log')
number_of_classes = 3
language = 'Tamil'
data_type = DataSetType.one_hot_encoding_character
base=''


def run_lstm_branch_approach(number_of_classes=number_of_classes, language=language,
                             data_type=DataSetType.sequence_word,
                             nb_epoch=200,
                             batch_size=128, ngram_range=(1, 1), max_length=20):
    logging.info("started loading")
    data_set = load_keras_train_data_set(language, number_of_classes,  max_length=max_length)
    logging.info("Loaded datasets")
    vocab = data_set['vocabulary']
    np.save('trained_models/temp/vocab.best.{}.Task{}'.format(language, number_of_classes-1), np.array(vocab))
    with open('trained_models/temp/maxlen.{}.Task{}.txt'.format(language, number_of_classes-1), "w") as max_file:
        max_file.write(str(max_length))
    logging.info("saved vocab and maxlen")

    length_input_layer = len(data_set['vocabulary'])
    model = models.lstm_branch_approach(vocabulary_size=len(data_set['vocabulary']),
                                        sequence_length=data_set['max_length'],
                                        number_of_classes=number_of_classes)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer='adadelta')
    logging.info("finished compile")
    checkpoint = ModelCheckpoint(base+"trained_models/temp/weights.best.{}.Task{}.hdf5".format(language,
                                                                                               number_of_classes-1),
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    early_stop = EarlyStopping(monitor='val_acc', patience=20, verbose=1)
    logging.info("staring fit")
    model.fit(data_set['X_train'], data_set['y_train'], callbacks=[checkpoint], verbose=1,
              validation_data=(data_set['X_test'], data_set['y_test_categorical']),
              nb_epoch=nb_epoch, batch_size=batch_size)
    avg_f1 = evaluate_keras_predictions(data_set['y_test'], model.predict(data_set['X_test']))
    print(model.evaluate(data_set['X_test'], data_set['y_test_categorical'],
                         batch_size=batch_size))
    result_logger.log_result(locals())
    return model


def predict(model, X_test, weights_path):
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer='adadelta')
    return model.predict(X_test)

def evaluate(y_pred, y_test):
    evaluate_keras_predictions(y_test, y_pred)

def predict_language_task(language, task):
    weights_path = "trained_models/temp/weights.best.{}.Task{}.hdf5".format(language, task)
    vocabs_path = "trained_models/temp/vocab.best.{}.Task{}.npy".format(language, task)
    with open("trained_models/temp/maxlen.{}.Task{}.txt".format(language, task)) as max_file:
        for line in max_file:
            max_length = int(line)
    vocab = list(np.load(vocabs_path))
    logging.info(max_length)
    dataset = load_keras_test_data_set(language, task+1, vocab, max_length)
    model = models.lstm_branch_approach(vocabulary_size=len(vocab),
                                        sequence_length=max_length,
                                        number_of_classes=task+1)
    predict(model, dataset['X'], weights_path)
    logging.info("Finished prediction")


# model = run_lstm_branch_approach(nb_epoch=10)
# model = run_simple_merge_approach(nb_epoch=10)
# model.save('/var/www/trained_models/temp/{}.model'.format(time.strftime("%Y%m%d_%H%M%S")))

task = 1
run_lstm_branch_approach(language='Malayalam', number_of_classes=2, nb_epoch=100, max_length=10)
predict_language_task('Malayalam', task)
#model = run_lstm_branch_approach(language='Malayalam', number_of_classes=3, nb_epoch=2, base=base)
#model = run_lstm_branch_approach(language='Punjabi', number_of_classes=2, nb_epoch=20, base=base)
#model = run_lstm_branch_approach(language='Punjabi', number_of_classes=3, nb_epoch=20, base=base)
# model = run_lstm_branch_approach(language='Hindi', number_of_classes=2, nb_epoch=20, base=base)
# model = run_lstm_branch_approach(language='Hindi', number_of_classes=3, nb_epoch=20, base=base)
#model = run_lstm_branch_approach(language='Tamil', number_of_classes=2, nb_epoch=1, base=base)
#model = run_lstm_branch_approach(language='Tamil', number_of_classes=3, nb_epoch=40, base=base)
# model = run_deep_dense_model()
