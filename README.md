# Paradox: An End-to-End System for Paraphrase Detection


## Corpus Statistics

Number of sentences in the training corpora

|     | Task 1 | Task 2 |
|:---:|:------:|:------:|
| Malayalam |  2500 | 3500  |
| Tamil | 2500  | 3500  |
| Hindi | 2500  | 3500  |
| Punjabi | 1700  | 2200  |


Number of training instances per classes in the training corpora

|     | Paraphrase | Not Paraphrase | SP |
|:---:|:------:|:------:|:------:|
| Malayalam Task 1 |  1000 | 1500  | 0 |
| Malayalam Task 2 |  1000 | 1500  | 1000 |
| Tamil Task 1 |  1000 | 1500  | 0 |
| Tamil Task 2 |  1000 | 1500  | 1000 |
| Hindi Task 1 |  1000 | 1500  | 0 |
| Hindi Task 2 |  1000 | 1500  | 1000 |
| Punjabi Task 1 |  700 | 1000  | 0 |
| Punjabi Task 2 |  700 | 1000  | 500 |


## Results as Average F1 Measure

a) Baseline CountVectorizer mit default Tokenizer, SVM Gridearch
b) Baseline CountVectorizer mit Hindi Tokenizer, SVM Gridearch

| Method    | Mal T1 | Mal T2 |
|:---:|:------:|:------:|
| a) |  0.697 |  0.581 |
| b) |   |   |

