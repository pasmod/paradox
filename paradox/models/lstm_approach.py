from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution1D, LSTM, Embedding


def get_model(length_input_layer, number_of_classes):
    model = Sequential()
    len_output_dim = 128
    max_len = 210
    model.add(Embedding(input_dim=length_input_layer, output_dim=len_output_dim, input_length=max_len, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
    #model.add(Convolution1D(64, 1, border_mode='same', input_shape=(1, length_input_layer)))
    # model.add(LSTM(128, input_shape=(40, length_input_layer)))
    # model.add(LSTM(128, input_dim=length_input_layer))
    model.add(Dense(64))
    model.add(Dense(30))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    return model
