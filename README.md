# Paradox: Automatic Paraphrase Identification
Given two sentences, paradox returns a continuous valued similarity score
on a scale from 0 to 5, with 0 indicating that the semantics of the sentences are completely
independent and 5 signifying semantic equivalence.

# Glove Models
Paradox used Glove pre-trained Glove models. Download the Glove models from [here](http://nlp.stanford.edu/data/glove.6B.zip) and put the zip file in the root folder. The ```make download_models``` (dicussed in the next section) will take care of the rest. 

# How to install:
Paradox is dockerized! First install Docker and then run the following commands:
```bash
cd paradox
make install
make download_models
```
the command ```make download_models``` assumes that you have already downloaded the Glove models.


# Training Corpus
For training, the semantic similarity corpora from SemEval (2012-2016) are used. The training data
are available under ```/corpus```.

# Required Models
Following models are required to run paradox:
- Glove: Download the models [here](http://nlp.stanford.edu/data/glove.6B.zip) and put them in the folder ```glove.6B```.

# Evaluation
To evaluate paradox using cross validation use:


# ToDos:
- Imlement topical similarity based of the LDA models.
