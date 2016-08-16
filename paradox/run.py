from paradox.parsers.corpus_parser import parse

X_malayalam_task1, y_malayalam_task1 = parse(path='../corpora/Malayalam/dpil-mal-train-Task1.xml')
X_malayalam_task2, y_malayalam_task2 = parse(path='../corpora/Malayalam/dpil-mal-train-Task2.xml')

X_tamil_task1, y_tamil_task1 = parse(path='../corpora/Tamil/dpil-tam-train-Task1.xml')
X_tamil_task2, y_tamil_task2 = parse(path='../corpora/Tamil/dpil-tam-train-Task2.xml')

X_hindi_task1, y_hindi_task1 = parse(path='../corpora/Hindi/dpil-hindi-train-Task1.xml')
X_hindi_task2, y_hindi_task2 = parse(path='../corpora/Hindi/dpil-hindi-train-Task2.xml')

X_punjabi_task1, y_punjabi_task1 = parse(path='../corpora/Punjabi/dpil-punjabi-train-Task1.xml')
X_punjabi_task2, y_punjabi_task2 = parse(path='../corpora/Punjabi/dpil-punjabi-train-Task2.xml')
