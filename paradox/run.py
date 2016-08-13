from paradox.parsers.corpus_parser import parse


X_malayalam_task1, y_malayalam_task1 = parse(path='../corpora/Malayalam/dpil-mal-train-Task1.xml')
X_malayalam_task2, y_malayalam_task2 = parse(path='../corpora/Malayalam/dpil-mal-train-Task2.xml')

X_tamil_task1, y_tamil_task1 = parse(path='../corpora/Tamil/dpil-tam-train-Task1.xml')
X_tamil_task2, y_tamil_task2 = parse(path='../corpora/Tamil/dpil-tam-train-Task2.xml')
