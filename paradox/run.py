# -*- coding: utf-8 -*-
from loaders.corpus_loader import load_all_languages
from evaluation.training_test_split import split_training_data
from evaluation.svm_baseline import estimate_svm_baseline
from pipelines.count_vectorizer_pipeline import create_count_vectorizer_pipeline
from pipelines.count_vectorizer_pipeline import create_char_count_vectorizer_pipeline
from utils.vocabulary_extractor import get_vocabulary
from tokenizers.hindi_tokenizer_wrapper import hindi_tokenize
import pipelines
import codecs
import sys

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)


def evaluate(test_train_split, method_name):
    if method_name == 'svm_baseline_default_tokenizer':
        pipeline = create_count_vectorizer_pipeline()
        estimate_svm_baseline(pipeline, test_train_split)
    elif method_name == 'svm_baseline_default_tokenizer_with_vocabulary':
        vocabulary = get_vocabulary(test_train_split['X_train'], tokenizer=None)
        pipeline = create_count_vectorizer_pipeline(vocabulary=vocabulary)
        estimate_svm_baseline(pipeline, test_train_split)
    elif method_name == 'svm_baseline_hindi_tokenizer':
        pipeline = create_count_vectorizer_pipeline(tokenizer=hindi_tokenize)
        estimate_svm_baseline(pipeline, test_train_split)
    elif method_name == 'svm_baseline_hindi_tokenizer_with_vocabulary':
        vocabulary = get_vocabulary(test_train_split['X_train'], tokenizer=hindi_tokenize)
        pipeline = create_count_vectorizer_pipeline(vocabulary=vocabulary, tokenizer=hindi_tokenize)
        estimate_svm_baseline(pipeline, test_train_split)
    elif method_name == 'svm_baseline_character_count_vectorizer':
        pipeline = create_char_count_vectorizer_pipeline()
        estimate_svm_baseline(pipeline, test_train_split)
    elif method_name == 'svm_baseline_character_count_vectorizer_with_vocabulary':
        vocabulary = get_vocabulary(test_train_split['X_train'], analyzer='char_wb')
        pipeline = create_char_count_vectorizer_pipeline(vocabulary=vocabulary)
        estimate_svm_baseline(pipeline, test_train_split)


data_sets = load_all_languages()
evaluation_method_name = 'svm_baseline_default_tokenizer_with_vocabulary'
for language, language_data_set in data_sets.iteritems():
    print 'Evaluating {}'.format(language)
    for x in range(1, 3):
        key = 'Task{}'.format(x)
        print('>> ' + key)
        test_train_split = split_training_data(language_data_set[key][0], language_data_set[key][1])
        evaluate(test_train_split, evaluation_method_name)
    print ''
