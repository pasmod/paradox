# Paradox: Automatic Paraphrase Identification
Given two sentences, paradox returns a continuous valued similarity score
on a scale from 0 to 5, with 0 indicating that the semantics of the sentences are completely
independent and 5 signifying semantic equivalence.

# Training Corpus
For training, the semantic similarity corpora from SemEval (2012-2016) are used. The training data
are available under ```/corpus```.

# Required Models
Following models are required to run paradox:
- Glove: Download the models [here](http://nlp.stanford.edu/data/glove.6B.zip) and put them in the folder ```glove.6B```.

# Evaluation
To evaluate paradox using cross validation use:



