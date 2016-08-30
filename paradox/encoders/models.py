from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution1D, LSTM, Embedding, Merge


def simple_model(length_input_layer, number_of_classes):
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
    model.add(Dense(number_of_classes, activation='softmax'))
    return model


def deep_dense_model(length_input_layer, number_of_classes):
    model = Sequential()
    model.add(Dense(128, input_dim=length_input_layer, init='uniform'))
    model.add(Activation('tanh'))
    # model.add(Dropout(0.5))
    model.add(Dense(100, init='uniform'))
    model.add(Activation('tanh'))
    # model.add(Dropout(0.5))
    model.add(Dense(64, init='uniform'))
    model.add(Activation('tanh'))
    # model.add(Dropout(0.5))
    model.add(Dense(30, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dense(10, init='uniform'))
    model.add(Dense(number_of_classes, activation='softmax'))
    return model


def simple_merge_approach(length_input_layer, number_of_classes):
    left_branch = Sequential()
    left_branch.add(Dense(64, input_dim=length_input_layer, init='uniform'))
    left_branch.add(Activation('tanh'))
    left_branch.add(Dropout(0.5))
    right_branch = Sequential()
    right_branch.add(Dense(64, input_dim=length_input_layer, init='uniform'))
    right_branch.add(Activation('tanh'))
    right_branch.add(Dropout(0.5))
    merged = Merge([left_branch, right_branch], mode='concat')
    model = Sequential()
    model.add(merged)
    model.add(Dense(32, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(10, init='uniform'))
    model.add(Dense(number_of_classes, activation='softmax'))
    return model


def lstm_approach(vocabulary_size, sequence_length, number_of_classes):
    model = Sequential()
    len_output_dim = 128
    max_len = 210
    model.add(Embedding(input_dim=vocabulary_size, output_dim=len_output_dim, input_length=sequence_length, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
    # model.add(Convolution1D(64, 1, border_mode='same', input_shape=(1, length_input_layer)))
    # model.add(LSTM(128, input_shape=(40, length_input_layer)))
    # model.add(LSTM(128, input_dim=length_input_layer))
    model.add(Dense(64))
    model.add(Dense(30))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    return model
