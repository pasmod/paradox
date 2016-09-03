from paradox.parsers.corpus_parser import parse
import os


def load_data_set(language, task_number, corpus_type=None):
    if corpus_type == 'train':
        corpus = 'corpora/{}/dpil-{}-train-Task{}.xml'
    else:
        corpus = 'corpora/{}/Test{}_Task{}.xml'
    path = corpus.format(language,
                         language,
                         task_number)
    return parse(path)


def load_all_languages(corpus_type='train'):
    languages = {'Malayalam', 'Tamil', 'Hindi', 'Punjabi'}
    data_sets = {}
    for language in languages:
        language_dict = {}
        language_dict['Task1'] = load_data_set(language, 1, corpus_type=corpus_type)
        language_dict['Task2'] = load_data_set(language, 2, corpus_type=corpus_type)
        data_sets[language] = language_dict
    return data_sets
