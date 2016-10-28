from pythonrouge import pythonrouge


def surface(text1, text2, method='ROUGE-2'):
    methods = ['ROUGE-1', 'ROUGE-2', 'ROUGE-3', 'ROUGE-SU4', 'ROUGE-L']
    if method not in methods:
        raise ValueError("Method {} is not supported."
                         "Available methods: {}".format(method, methods))
    return pythonrouge.pythonrouge(text1, text2)[method]
