sp2016.11-731
=============

Master repository for CMU Machine Translation (11-731) in Spring 2016

# Posted Assignments

 * Homework 0
 * Homework 1

=======
There are three Python programs here (`-h` for usage):

 - `./evaluate` evaluates pairs of MT output hypotheses relative to a reference translation using counts of matched words
 - `./check` checks that the output file is correctly formatted
 - `./grade` computes the accuracy

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./evaluate | ./check | ./grade


The `data/` directory contains the following two files:

 - `data/train-test.hyp1-hyp2-ref` is a file containing tuples of two translation hypotheses and a human (gold standard) translation. The first 26208 tuples are training data. The remaining 24131 tuples are test data.

 - `data/train.gold` contains gold standard human judgements indicating whether the first hypothesis (hyp1) or the second hypothesis (hyp2) is better or equally good/bad for training data.

Until the deadline the scores shown on the leaderboard will be accuracy on the training set. After the deadline, scores on the blind test set will be revealed and used for final grading of the assignment.

=======

Strategies:
For our evaluator, we built a logistic regression classifier using the following features from the data (tokenized and lowercased in cdec):
 - Pre-trained 50-dimension GloVe vector representations:  We averaged the minimum cosine similarity between each word in a hypothesis and a word in the reference
translation.
 - Socher et al (2011) Dynamic Pooling And Unfolding Recursive Autoencoders For Paraphrase Detection:  We learned a vector representation for each sentence in the
two hypotheses and the reference and took the cosine similarity between a hypothesis vector and the reference vector.
 - Language model: We used KenLM to build an order 3 language model trained on FR-EN Europarl English sentences.  The log probability of a hypothesis was used as
a feature.
 - Simple METEOR: Harmonic mean of precision and recall of both word and unigram POS matches between a hypothesis and a reference.
 - BLEU-4: BLEU score up to 4-grams with brevity penalty.
 - Sentence length: Length of each hypothesis and the reference.

The original plan for the evaluator was to build a neural evaluator based on Baltescu & Blunsom (2015) with an alternate method for combining semantic vectors
in a sentence.  However, due to concerns about overfitting, we are submitting the logistic regression evaluator for the assignment (Please disregard Dan, he's
being silly :P).
