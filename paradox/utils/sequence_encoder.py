import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.feature_extraction.text import CountVectorizer
from paradox.tokenizers.hindi_tokenizer_wrapper import hindi_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def _encode_sequence(X, max_length, vocabulary, tokenizer=hindi_tokenize, concat_vectors=True):
    tokenized_sentences_A = [(tokenizer(x[0])) for x in X]
    tokenized_sentences_B = [(tokenizer(x[1])) for x in X]
    encodings_sentences_A = []
    encodings_sentences_B = []
    padded_A = pad(tokenized_sentences_A, max_length)
    padded_B = pad(tokenized_sentences_B, max_length)
    vocabulary = {x: i for i, x in enumerate(vocabulary)}
    for i in range(len(padded_A)):
        s1 = [vocabulary.get(word, vocabulary.get('<UNK/>')) for word in padded_A[i]]
        s2 = [vocabulary.get(word, vocabulary.get('<UNK/>')) for word in padded_B[i]]
        encodings_sentences_A.append(s1)
        encodings_sentences_B.append(s2)
    encodings_sentences_A = np.array(encodings_sentences_A)
    encodings_sentences_B = np.array(encodings_sentences_B)
    if concat_vectors:
        return np.concatenate((encodings_sentences_A, encodings_sentences_B), axis=1)
    else:
        return [encodings_sentences_A, encodings_sentences_B]


def encode_sequence(X, max_length, vocabulary, tokenizer=hindi_tokenize, concat_vectors=True):
    t = Tokenizer()
    vectorizer = CountVectorizer(analyzer='word', vocabulary=vocabulary, binary=True)
    sentences_A = [x[0] for x in X]
    sentences_B = [x[1] for x in X]
    #sentences_A = [x[0].encode("utf-8") for x in X]
    #sentences_B = [x[1].encode("utf-8") for x in X]
    encodings_sentences_A = [s.split(" ") for s in sentences_A]
    encodings_sentences_B = [s.split(" ") for s in sentences_B]
    padded_A = pad(encodings_sentences_A, max_length)
    padded_B = pad(encodings_sentences_B, max_length)
    vocabulary = {x: i for i, x in enumerate(vocabulary)}

    X_train_A = np.zeros((len(sentences_A), max_length, len(vocabulary)), dtype=np.bool)
    X_train_B = np.zeros((len(sentences_B), max_length, len(vocabulary)), dtype=np.bool)
    for i, sentence in enumerate(padded_A):
        for t, word in enumerate(sentence):
            X_train_A[i, t, vocabulary.get(word, vocabulary.get('<UNK/>'))] = 1
    for i, sentence in enumerate(padded_B):
        for t, word in enumerate(sentence):
            X_train_B[i, t, vocabulary.get(word, vocabulary.get('<UNK/>'))] = 1

    if concat_vectors:
        return np.concatenate((X_train_A, X_train_B), axis=1)
    else:
        return [X_train_A, X_train_B]


def pad(X, max_length, padding_word="<PAD/>"):
    padded_sentences = []
    for i in range(len(X)):
        s = X[i][:max_length]
        num_padding = max_length - len(s)
        new_sentence = []
        new_sentence.extend(s)
        for i in range(num_padding):
            new_sentence.append(padding_word)
        padded_sentences.append(new_sentence)
    return padded_sentences
