from paradox.loaders.corpus_loader import load_all_languages
from paradox.evaluation.training_test_split import split_training_data
from paradox.utils.vocabulary_extractor import get_vocabulary
from paradox.utils.one_hot_encoder import get_one_hot_encoding
from keras.utils.np_utils import to_categorical
from enum import Enum
import numpy as np
from paradox.utils.sequence_encoder import encode_sequence, pad
from paradox.tokenizers.hindi_tokenizer_wrapper import hindi_tokenize


def load_keras_train_data_set(language, number_of_classes, max_length):
    task = 'Task1'
    if number_of_classes == 3:
        task = 'Task2'
    data_sets = load_all_languages()
    test_train_split = split_training_data(data_sets[language][task][0], data_sets[language][task][1])
    analyzer = 'word'
    vocabulary = get_vocabulary(test_train_split['X_train'])
    vocabulary.append("<PAD/>")
    vocabulary.append("<UNK/>")
    X_train = encode_sequence(test_train_split['X_train'], max_length, vocabulary)
    # print X_train.shape
    X_test = encode_sequence(test_train_split['X_test'], max_length, vocabulary)
    return {'X_train': X_train,
            'X_test': X_test,
            'y_train': to_categorical(np.array(test_train_split['y_train']), number_of_classes),
            'y_test_categorical': to_categorical(np.array(test_train_split['y_test']), number_of_classes),
            'y_test': test_train_split['y_test'],
            'vocabulary': vocabulary,
            'max_length': max_length}


class DataSetType(Enum):
    one_hot_encoding_word = 1
    one_hot_encoding_character = 2
    sequence_character = 3
    sequence_word = 4
