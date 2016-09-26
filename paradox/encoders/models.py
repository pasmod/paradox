from keras.layers import Dense, Embedding
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Merge


def lstm_encoder(encoding_dim=32, depth=3, activation='relu',
                 max_len=None, max_features=None, embeddings=None,
                 vocab=None, lang='en', weights=None):
    left_branch = Sequential()
    if lang == 'en':
        left_branch.add(Embedding(len(vocab) + 1, 200, input_length=max_len,
                                  weights=[embeddings],
                                  trainable=True))
    if lang == 'de':
        left_branch.add(Embedding(weights.shape[0], weights.shape[1],
                                  input_length=max_len,
                                  weights=[weights],
                                  trainable=True))
    right_branch = Sequential()
    if lang == 'en':
        right_branch.add(Embedding(len(vocab) + 1, 200, input_length=max_len,
                                   weights=[embeddings],
                                   trainable=True))
    if lang == 'de':
        right_branch.add(Embedding(weights.shape[0], weights.shape[1],
                                   input_length=max_len,
                                   weights=[weights],
                                   trainable=True))
    merged = Merge([left_branch, right_branch], mode='concat')
    model = Sequential()
    model.add(merged)
    model.add(LSTM(128, dropout_W=0.5, return_sequences=True))
    model.add(LSTM(64, dropout_W=0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
