#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

unsigned GLOVE_DIM = 50;
unsigned SENTENCE_DIM = 100;
unsigned INPUT_DIM = GLOVE_DIM * GLOVE_DIM + SENTENCE_DIM;
unsigned HIDDEN_DIM = 45;
unsigned PAIRWISE_DIM = 100;
unsigned OUTPUT_DIM = 1;

class Sentence {
  public:
    vector<float> syn;
    vector<vector<float>> sem;
    float BLEU;
    float meteor;
};

class Instance {
  public:
    Sentence hyp1;
    Sentence hyp2;
    Sentence ref;
    int correct;
};

Expression buildComputationGraph(Instance instance,
 ComputationGraph& cg, Model m) {
  Expression input_embed = parameter(cg, m.add_parameters({HIDDEN_DIM, INPUT_DIM}));
  Expression W12 = parameter(cg, m.add_parameters({PAIRWISE_DIM, HIDDEN_DIM * 2}));
  Expression b12 = parameter(cg, m.add_parameters({PAIRWISE_DIM}));
  Expression W1r = parameter(cg, m.add_parameters({PAIRWISE_DIM, HIDDEN_DIM * 2}));
  Expression b1r = parameter(cg, m.add_parameters({PAIRWISE_DIM}));
  Expression W2r = parameter(cg, m.add_parameters({PAIRWISE_DIM, HIDDEN_DIM * 2}));
  Expression b2r = parameter(cg, m.add_parameters({PAIRWISE_DIM}));
  Expression V = parameter(cg, m.add_parameters({1, PAIRWISE_DIM * 3 + 4}));
  Expression b = parameter(cg, m.add_parameters({1}));

  // Create embedding from syntax and semantic vector
  Expression hyp1syn = input(cg, {SENTENCE_DIM, 1}, instance.hyp1.syn);
  Expression hyp2syn = input(cg, {SENTENCE_DIM, 1}, instance.hyp2.syn);
  Expression refsyn = input(cg, {SENTENCE_DIM, 1}, instance.ref.syn);

  vector<Expression> hyp1sem_vectors;
  vector<Expression> hyp2sem_vectors;
  vector<Expression> refsem_vectors;
  for (int i = 0; i < instance.hyp1.sem.size(); ++i) {
    hyp1sem_vectors.push_back(input(cg, {GLOVE_DIM, 1}, instance.hyp1.sem[i]));
    hyp2sem_vectors.push_back(input(cg, {GLOVE_DIM, 1}, instance.hyp2.sem[i]));
    refsem_vectors.push_back(input(cg, {GLOVE_DIM, 1}, instance.ref.sem[i])); 
  }
  Expression hyp1sem_matrix = concatenate_cols(hyp1sem_vectors);
  Expression hyp2sem_matrix = concatenate_cols(hyp2sem_vectors);
  Expression refsem_matrix = concatenate_cols(refsem_vectors);

  Expression hyp1sem = reshape(hyp1sem_matrix * transpose(hyp1sem_matrix), {GLOVE_DIM * GLOVE_DIM, 1});
  Expression hyp2sem = reshape(hyp2sem_matrix * transpose(hyp2sem_matrix), {GLOVE_DIM * GLOVE_DIM, 1});
  Expression refsem = reshape(refsem_matrix * transpose(refsem_matrix), {GLOVE_DIM * GLOVE_DIM, 1});

  Expression x1 = input_embed * concatenate({hyp1syn, hyp1sem});
  Expression x2 = input_embed * concatenate({hyp2syn, hyp2sem});
  Expression xref = input_embed * concatenate({refsyn, refsem});

  // Pairwise vectors
  Expression h12 = tanh(W12 * concatenate({x1, x2}) + b12);
  Expression h1r = tanh(W1r * concatenate({x1, xref}) + b1r);
  Expression h2r = tanh(W2r * concatenate({x2, xref}) + b2r);

  // Combination of evaluation input
  Expression BLEU1 = input(cg, instance.hyp1.BLEU);
  Expression BLEU2 = input(cg, instance.hyp2.BLEU);
  Expression meteor1 = input(cg, instance.hyp1.meteor);
  Expression meteor2 = input(cg, instance.hyp2.meteor);
  Expression combined = concatenate({h12, h1r, h2r, BLEU1, BLEU2, meteor1, meteor2});

  Expression y_pred = logistic(V * combined + b);
  Expression y = input(cg, instance.correct);
  Expression loss = binary_log_loss(y_pred, y);

  return loss;
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  bool isTrain = true;

}
