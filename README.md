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

a) Baseline CountVectorizer mit default Tokenizer, SVM Gridsearch
b) Baseline CountVectorizer mit Hindi Tokenizer, SVM Gridsearch

| Method    | Malayalam T1 | Malayalam T2 | Punjabi T1 | Punjabi T2 | Hindi T1 | Hindi T2 | Tamil T1 | Tamil T2 |
|:---:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| a) |  0.697  |  0.581 | 0.414 | 0.329| 0.411 | 0.496 | 0.825 | 0.626 |
| b) |  0.7157 |  0.690 | 0.39  | 0.25 | 0.417 | | 0.845 | 0.613 |

