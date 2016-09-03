from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution1D, LSTM, Embedding, Merge


def lstm_branch_approach(vocabulary_size, sequence_length, number_of_classes):
    left_branch = Sequential()
    left_branch.add(LSTM(128, input_shape=(sequence_length, vocabulary_size),
                         return_sequences=True))
    left_branch.add(LSTM(64, return_sequences=True))
    right_branch = Sequential()
    right_branch.add(LSTM(128, input_shape=(sequence_length, vocabulary_size),
                          return_sequences=True))
    right_branch.add(LSTM(64, return_sequences=True))
    merged = Merge([left_branch, right_branch], mode='concat')
    model = Sequential()
    model.add(merged)
    model.add(LSTM(64, dropout_W=0.8, return_sequences=True))
    model.add(LSTM(32, dropout_W=0.8))
    model.add(Dense(32))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    return model
