# -*- coding: utf-8 -*-
from parsers.corpus_parser import parse
from evaluation.training_test_split import split_training_data
from evaluation.count_vectorizer_word_baseline import estimate_svm_baseline
from tokenizers.hindi_tokenizer import Tokenizer

import codecs
import sys

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

X_malayalam_task1, y_malayalam_task1 = parse(path='../corpora/Malayalam/dpil-mal-train-Task1.xml')
X_malayalam_task2, y_malayalam_task2 = parse(path='../corpora/Malayalam/dpil-mal-train-Task2.xml')

X_tamil_task1, y_tamil_task1 = parse(path='../corpora/Tamil/dpil-tam-train-Task1.xml')
X_tamil_task2, y_tamil_task2 = parse(path='../corpora/Tamil/dpil-tam-train-Task2.xml')

X_hindi_task1, y_hindi_task1 = parse(path='../corpora/Hindi/dpil-hindi-train-Task1.xml')
X_hindi_task2, y_hindi_task2 = parse(path='../corpora/Hindi/dpil-hindi-train-Task2.xml')

X_punjabi_task1, y_punjabi_task1 = parse(path='../corpora/Punjabi/dpil-punjabi-train-Task1.xml')
X_punjabi_task2, y_punjabi_task2 = parse(path='../corpora/Punjabi/dpil-punjabi-train-Task2.xml')

test_train_split_malayalam_task1 = split_training_data(X_malayalam_task1, y_malayalam_task1)
estimate_svm_baseline(test_train_split_malayalam_task1)
