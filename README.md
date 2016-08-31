# Paradox: An End-to-End System for Paraphrase Detection


## Corpus Statistics

Number of training instances per classes in the training corpora

|     | Paraphrase | Not Paraphrase | SP | Total
|:---:|:------:|:------:|:------:|:------:|
| Malayalam Task 1 |  1000 | 1500  | 0 | 2500 |
| Malayalam Task 2 |  1000 | 1500  | 1000 | 3500 |
| Tamil Task 1 |  1000 | 1500  | 0 | 2500 |
| Tamil Task 2 |  1000 | 1500  | 1000 | 3500 |
| Hindi Task 1 |  1000 | 1500  | 0 | 2500 |
| Hindi Task 2 |  1000 | 1500  | 1000 | 3500 |
| Punjabi Task 1 |  700 | 1000  | 0 | 1700 |
| Punjabi Task 2 |  700 | 1000  | 500 | 2200 |


## Results as Average F1 Measure

<ol type="a">
    <li>Baseline CountVectorizer with default Tokenizer, SVM Gridsearch</li>
    <li>Baseline CountVectorizer with default Tokenizer, SVM Gridsearch, shared vocabulary</li>
    <li>Baseline CountVectorizer with Hindi Tokenizer, SVM Gridsearch</li>
    <li>Baseline CountVectorizer with Hindi Tokenizer, SVM Gridsearch, shared vocabulary</li>
    <li>Baseline CountVectorizer with character analyzer</li>
    <li>Baseline CountVectorizer with character analyzer, shared vocabulary</li>
    <li>run_lstm_branch_approach (trim 30 words per sentence), 10 epochs, batch_size 2, dropout=0.2</li>
<ol>

| Method    | Malayalam T1 | Malayalam T2 | Punjabi T1 | Punjabi T2 | Hindi T1 | Hindi T2 | Tamil T1 | Tamil T2 |
|:---:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| a) |  0.697  |  0.581 | 0.414 | 0.329 | 0.411 | 0.496 | 0.825 | 0.626 |
| b) |         |        |       |       |       |       |       |       |
| c) |  0.715  |  0.690 | 0.390 | 0.250 | 0.417 | 0.440 | 0.845 | 0.613 |
| d) |         |        |       |       |       |       |       |       |
| e) |  0.741  |  0.721 | 0.815 | 0.768 | 0.724 | 0.663 | 0.840 | 0.700 |
| f) |  0.741  |  0.721 | 0.815 | 0.768 | 0.724 | 0.663 | 0.840 | 0.700 |
| g) |  0.636  |  0.589 | 0.487 | 0.376 | 0.480 | 0.484 | 0.805 | 0.604 |

