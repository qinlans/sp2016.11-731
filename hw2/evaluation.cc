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

unsigned INPUT_DIM = 0;
unsigned HIDDEN_DIM = 45;
unsigned PAIRWISE_DIM = 100;
unsigned OUTPUT_DIM = 1;

class Sentence {
  public:
    Expression syn;
    Expression sem;
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
  Expression x1 = input_embed * concatenate({instance.hyp1.syn, instance.hyp1.sem});
  Expression x2 = input_embed * concatenate({instance.hyp2.syn, instance.hyp2.sem});
  Expression xref = input_embed * concatenate({instance.ref.syn, instance.ref.sem});

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
