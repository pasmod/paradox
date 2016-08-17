from parsers.corpus_parser import parse


def load_data_set(language, task_number):
    return parse(path='../corpora/{}/dpil-{}-train-Task{}.xml'.format(language, language, task_number))


def load_all_languages():
    languages = {'Malayalam', 'Tamil', 'Hindi', 'Punjabi'}
    data_sets = {}
    for language in languages:
        language_dict = {}
        language_dict['Task1'] = load_data_set(language, 1)
        language_dict['Task2'] = load_data_set(language, 2)
        data_sets[language] = language_dict
    return data_sets
