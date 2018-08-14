# Paradox: Automatic Paraphrase Identification

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/pasmod/paradox/blob/master/License.md)

Given two sentences, paradox returns a continuous valued similarity score
on a scale from 0 to 5, with 0 indicating that the semantics of the sentences are completely
independent and 5 signifying semantic equivalence. Paradox uses [Glove](https://nlp.stanford.edu/projects/glove/) [pre-trained models](http://nlp.stanford.edu/data/glove.6B.zip).

## How to install

Paradox is dockerized! First install Docker and then run the following commands:

```bash
cd paradox
make install
make download_glove
make download_models
```

## Training Corpus

For training, the semantic similarity corpora from SemEval (2012-2016) are used. The training data
are available under ```/corpus```.

## Evaluation

The evaluation scipt reports the results on the test data set of the SemEval2016 challange. To see
the resport run the following commands:
```bash
source env/bin/activate
python paradox/benchmark.py
```

## Citation
This repository contains the code for the DeepLDA approach introduced in the following paper. Use the following bibtex entry to cite us:
``` bash
@InProceedings{liebeck-EtAl:2016:SemEval,
    author    = {Liebeck, Matthias and Pollack, Philipp and Modaresi, Pashutan and Conrad, Stefan},
    title     = {HHU at SemEval-2016 Task 1: Multiple Approaches to Measuring Semantic Textual Similarity},
    booktitle = {Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016)},
    month     = {June},
    year      = {2016},
    address   = {San Diego, California},
    publisher = {Association for Computational Linguistics},
    pages     = {607--613},
    url       = {TOBEFILLED-http://www.aclweb.org/anthology/W/W05/W05-0292}
}
```

## ToDos:
- Implement topical similarity based of the LDA models.
