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

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
}
