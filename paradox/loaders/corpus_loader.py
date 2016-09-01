from paradox.parsers.corpus_parser import parse
import os


def load_data_set(language, task_number, base=''):
    corpus = 'corpora/{}/dpil-{}-train-Task{}.xml'
    path = os.path.join(base, corpus.format(language,
                                            language,
                                            task_number))
    return parse(path)


def load_all_languages(base=''):
    languages = {'Malayalam', 'Tamil', 'Hindi', 'Punjabi'}
    data_sets = {}
    for language in languages:
        language_dict = {}
        language_dict['Task1'] = load_data_set(language, 1, base=base)
        language_dict['Task2'] = load_data_set(language, 2, base=base)
        data_sets[language] = language_dict
    return data_sets
