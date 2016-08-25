from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation


def get_model(length_input_layer, number_of_classes):
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
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
