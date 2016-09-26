from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from collections import Counter
import numpy as np
from keras.utils import np_utils
from parser import parse
import logging
import itertools


def word2vec(word_index):
    import os
    import numpy as np
    embeddings_index = {}
    f = open(os.path.join("embeddings", 'glove.6B.200d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((len(word_index) + 1, 200))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def word2vec_embedding_layer(embeddings_path):
    return np.load(open(embeddings_path, 'rb'))


def load(lang=None, n=1000, test_size=0.2):
    """Loads n instances of the corpus in a form
    suitable to be used by keras models.

    # Arguments
        n: number of required instances
        test_size: ratio of data to be used for testing

    # Returns:
        (X1_train, X2_train, y_train), (X1_test, X2_test, y_test)
    """
    X, Y = _load(lang=lang, n=n)
    padded_X = pad(X)
    vocab = build_vocab(padded_X)
    embeddings = None
    weights = None
    embeddings = word2vec(vocab)
    weights = word2vec_embedding_layer('embeddings/word2vec_wiki-de_200_binary.syn0.npy')
    x, y = build_input_data(padded_X, Y, vocab)
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_size,
                                                        stratify=y,
                                                        random_state=42)
    return (np.array([xx[0] for xx in X_train]),
            np.array([xx[1] for xx in X_train]),
            y_train), (np.array([xx[0] for xx in X_test]),
                       np.array([xx[1] for xx in X_test]),
                       y_test), vocab, embeddings, weights


def encode_labels(Y):
    """Encodes the labels.

    # Arguments
        Y: a list of the form [True, False, ...]

    # Returns:
        a list of the form [[1, 0], [0, 1],...]
    """
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    return np_utils.to_categorical(encoded_Y)
    return encoded_Y


def build_vocab(X):
    """Builds the vocabulary index mapping each
    token to an index.

    # Arguments
        X: a list of tuples of tokenizes sentences.
           Example: [(["t1", "t2"], ["t3"]),
           (["t3"], ["t4", "t5"]), ...]

    # Returns:
        a map between tokens and their corresponding index
    """
    word_counts = Counter(itertools.chain(*[x[0] + x[1] for x in X]))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return vocabulary


def build_input_data(X, Y, vocabulary):
    """Maps inputs and labels to vectors based
    on the vocabulary.

    # Arguments
        X: a list of tuples of padded tokenizes sentences.
           Example: [(["t1", "t2", "<PAD/>"], ["t3", "<PAD/>", "<PAD/>"]),
           (["t3", "</PAD>", "<PAD/>"], ["t4", "t5", "<PAD/>"]), ...]
        Y: a list of labels
                Example: [True, False, ...]
        vocabulary: a map between tokens and their indices

    # Returns:
        x, y: vector representation of inputs and labels
    """
    X_vec = []
    for x in X:
        s1 = [vocabulary[word] for word in x[0]]
        s2 = [vocabulary[word] for word in x[1]]
        X_vec.append((s1, s2))
    Y = encode_labels(Y)
    return np.array(X_vec), np.array(Y)


def pad(X, padding_word="<PAD/>"):
    """Pads the inputs to till the max length of
    inputs is reached.

    # Arguments
        X: a list of tuples of tokenized sentences.
           Example: [(["t1", "t2"], ["t3"]),
           (["t3"], ["t4", "t5"]), ...]

    # Returns:
        a list of tuples of padded tokenized sentences
        Example: [(["t1", "t2", "<PAD/>"], ["t3", "<PAD/>", "<PAD/>"]),
                  (["t3", "</PAD>", "<PAD/>"], ["t4", "t5", "<PAD/>"]), ...]
    """
    max_length = max(max(len(x1), len(x2)) for x1, x2 in X)
    padded_sentences = []
    for i in range(len(X)):
        pair = []
        for s in X[i]:
            num_padding = max_length - len(s)
            new_sentence = s + [padding_word] * num_padding
            pair.append(new_sentence)
        padded_sentences.append(tuple(pair))
    return padded_sentences


def _load(lang=None, n=None):
    count = 0
    instances = []
    for instance in parse(lang=lang):
        instances.append(instance)
        count = count + 1
        if count == n:
            break
    logging.info("Loaded {} sentence pairs".format(n))
    X = [(tokenize(instance[0]),
          tokenize(instance[1])) for instance in instances]
    Y = [instance[2] for instance in instances]
    return X, Y


def tokenize(text):
    return text.split(" ")
