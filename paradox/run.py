# -*- coding: utf-8 -*-
from loaders.corpus_loader import load_all_languages
from evaluation.training_test_split import split_training_data
from evaluation.count_vectorizer_word_baseline import estimate_svm_baseline
import codecs
import sys

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

data_sets = load_all_languages()

for language, language_data_set in data_sets.iteritems():
    print 'Evaluating {}'.format(language)
    for x in range(1, 2):
        key = 'Task{}'.format(x)
        print('>> ' + key)
        test_train_split = split_training_data(language_data_set[key][0], language_data_set[key][1])
        estimate_svm_baseline(test_train_split)
    print ''
