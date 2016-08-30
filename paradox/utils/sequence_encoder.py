import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tokenizers.hindi_tokenizer_wrapper import hindi_tokenize


def encode_sequence(X, max_length, vocabulary, tokenizer=hindi_tokenize, concat_vectors=True):
    tokenized_sentences_A = [(tokenizer(x[0])) for x in X]
    tokenized_sentences_B = [(tokenizer(x[1])) for x in X]
    # for word in tokenized_sentences_A[0]:
        # print word.encode('utf-8')

    # print(len(tokenized_sentences_A[0]))

    encodings_sentences_A = []
    encodings_sentences_B = []

    padded_A = pad(tokenized_sentences_A, max_length)
    padded_B = pad(tokenized_sentences_B, max_length)

    # print type(vocabulary)
    vocabulary = {x: i for i, x in enumerate(vocabulary)}
    #print len(vocabulary)
    #for bla in vocabulary.keys():
        #print bla.encode('utf-8')
    #
    # print type(vocabulary)
    # print padded_A[0][0]
    # print type(padded_A[0][0])

    for i in range(len(padded_A)):
        #print padded_A[i][0].encode('utf-8')
        # print type(padded_A[i][0])
        # if padded_A[i][0] in vocabulary:
            # print 'in dict'
        # print padded_A[i],
        # print word
        # print type(word)
        # print padded_A[0]
        s1 = [vocabulary.get(word, vocabulary.get('<UNK/>')) for word in padded_A[i]]
        print s1
        s2 = [vocabulary.get(word, vocabulary.get('<UNK/>')) for word in padded_B[i]]
        encodings_sentences_A.append(s1)
        encodings_sentences_B.append(s2)
    encodings_sentences_A = np.array(encodings_sentences_A)
    encodings_sentences_B = np.array(encodings_sentences_B)

    if concat_vectors:
        return np.concatenate((encodings_sentences_A, encodings_sentences_B), axis=1)
    else:
        return [np.array(encodings_sentences_A), np.array(encodings_sentences_B)]

    X_vec = []
    # for x in X:
    # s1 = [vocabulary[word] for word in x[0]]
    # s2 = [vocabulary[word] for word in x[1]]
    # X_vec.append((s1, s2))
    # return np.array(X_vec)


def pad(X, max_length, padding_word="<PAD/>"):
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
    # max_length = max(max(len(x1), len(x2)) for x1, x2 in X)
    # print len(X)
    # print type(X[0])
    # print X[0]
    # print X
    padded_sentences = []
    for i in range(len(X)):
        s = X[i][:max_length]
        # print s
        num_padding = max_length - len(s)
        new_sentence = []
        new_sentence.extend(s)
        for i in range(num_padding):
            new_sentence.append(padding_word)
        #new_sentence = str(s.encode('utf-8')) + str([padding_word] * num_padding)
        padded_sentences.append(new_sentence)
    return padded_sentences
