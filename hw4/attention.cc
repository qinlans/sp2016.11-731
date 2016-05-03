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

unsigned LAYERS = 1;
unsigned OUT_LAYERS = 2;
unsigned ENC_INPUT_DIM = 40;
unsigned ENC_HIDDEN_DIM = 80;
unsigned DEC_INPUT_DIM = ENC_HIDDEN_DIM * 3;
unsigned DEC_HIDDEN_DIM = 80;
unsigned SRC_VOCAB_SIZE = 0;
unsigned TGT_VOCAB_SIZE = 0;
unsigned DROPOUT = .5;
bool LOAD_MODEL = false;
bool USE_DROPOUT = false;
float DROPOUT_RATE = .5;
bool TRAIN = true;

cnn::Dict src_d;
cnn::Dict tgt_d;

class Sentence {
  public:
    vector<string> words;
};

class Instance {
  public:
    Sentence source;
    Sentence target;
};

// Sets up the bidirectional LSTM for encoding a sentence
template <class Builder>
struct Encoder {
  LookupParameters* src_words;
  Builder l2rbuilder;
  Builder r2lbuilder;
  Parameters* p_bias; // One bias for both directions

  explicit Encoder(Model& model): 
      l2rbuilder(LAYERS, ENC_INPUT_DIM, ENC_HIDDEN_DIM, &model),
      r2lbuilder(LAYERS, ENC_INPUT_DIM, ENC_HIDDEN_DIM, &model) {
      src_words = model.add_lookup_parameters(SRC_VOCAB_SIZE, {ENC_INPUT_DIM});
      p_bias = model.add_parameters({ENC_HIDDEN_DIM*2});
  }

  // Builds the computation graphs and encodes a sentence
  vector<Expression> encode_sentence(ComputationGraph& cg, Sentence sentence) {
    l2rbuilder.new_graph(cg);
    if (USE_DROPOUT) {
      l2rbuilder.set_dropout(DROPOUT_RATE);
    }
    else {
      l2rbuilder.disable_dropout();
    }
    l2rbuilder.start_new_sequence();

    r2lbuilder.new_graph(cg);
    if (USE_DROPOUT) {
      r2lbuilder.set_dropout(DROPOUT_RATE);
    }
    else {
      r2lbuilder.disable_dropout();
    }
    r2lbuilder.start_new_sequence();

    Expression bias = parameter(cg, p_bias);

    vector<Expression> left_context;
    vector<Expression> right_context;
    vector<Expression> y;
    Expression y_i;

    vector<string> words = sentence.words;
    int sentenceLength = words.size();

    vector<unsigned> x;
    for (int i = 0; i < sentenceLength; ++i) {
      x.push_back(src_d.Convert(words[i]));
    }

    // Adds each word to the LSTM in both directions
    for (int i = 0; i < sentenceLength; ++i) {
      y_i = lookup(cg, src_words, x[i]);
      l2rbuilder.add_input(y_i);
      left_context.push_back(l2rbuilder.back());
    }
    for (int i = sentenceLength - 1; i >= 0; --i) {
      y_i = lookup(cg, src_words, x[i]);
      r2lbuilder.add_input(y_i);
      right_context.insert(right_context.begin(), r2lbuilder.back());
    }

    for (int i = 0; i < sentenceLength; ++i) {
      y.push_back(rectify(bias + concatenate({left_context[i], right_context[i]})));
    }

    return y;
  }
};

// Reads in sentences from given file
vector<Sentence> readData(string filename) {
  string line;
  string word;
  ifstream in(filename);
  vector<Sentence> sentences;

  while (getline(in, line)) {
    Sentence sentence;
    istringstream iss(line);

    vector<string> words;
    while (getline(iss, word, ' ')) {
      words.push_back(word);
    }
    sentence.words = words;
    sentences.push_back(sentence);
  }

  return sentences;
    
}

// Builds word dictionary
void buildDict(vector<Sentence> sentences, bool source) {
  for (int i = 0; i < sentences.size(); ++i) {
    for (int j = 0; j < sentences[i].words.size(); ++j) {
      if (source) {
        src_d.Convert(sentences[i].words[j]);
      }
      else {
        tgt_d.Convert(sentences[i].words[j]);
      }
    }
  }
}

// Builds instance vector
vector<Instance> buildInstances(vector<Sentence> src, vector<Sentence> tgt) {
  vector<Instance> instances;
  for (int i = 0; i < src.size(); ++i) {
    Instance instance;
    instance.source = src[i];
    instance.target = tgt[i];
    instances.push_back(instance);
  }
  return instances;
}

Expression position(ComputationGraph& cg, unsigned src_pos,
    unsigned tgt_pos, unsigned src_len) {
  Expression src_pos_e = input(cg, {1}, {1 + src_pos});
  Expression tgt_pos_e = input(cg, {1}, {1 + tgt_pos});
  Expression src_len_e = input(cg, {1}, {1 + src_len});

  return concatenate({log(src_pos_e), log(tgt_pos_e), log(src_len_e)});
}

template <class Builder>
struct Decoder {
  LookupParameters* tgt_words;
  Builder state_builder;
  Parameters* p_decode_start;
  Parameters* p_A;
  Parameters* p_R;
  Parameters* p_bias;
  explicit Decoder(Model& model) :
      state_builder(OUT_LAYERS, DEC_INPUT_DIM, DEC_HIDDEN_DIM, &model) {
      tgt_words = model.add_lookup_parameters(TGT_VOCAB_SIZE, {DEC_HIDDEN_DIM});
      p_decode_start = model.add_parameters({DEC_INPUT_DIM});
      p_A = model.add_parameters({ENC_HIDDEN_DIM * 2 + DEC_HIDDEN_DIM + 3});
      p_R = model.add_parameters({TGT_VOCAB_SIZE, DEC_HIDDEN_DIM});
      p_bias = model.add_parameters({TGT_VOCAB_SIZE});
  }

  Expression decode_train(ComputationGraph& cg, Sentence target, vector<Expression> encoded) {
    state_builder.new_graph(cg);
    if (USE_DROPOUT) {
      state_builder.set_dropout(DROPOUT_RATE);
    }
    else {
      state_builder.disable_dropout();
    }
    state_builder.start_new_sequence();

    Expression decode_start = parameter(cg, p_decode_start);
    Expression A = parameter(cg, p_A);
    Expression R = parameter(cg, p_R);
    Expression bias = parameter(cg, p_bias);
    vector<Expression> errs;

    state_builder.add_input(decode_start);
    for (int i = 0; i < target.words.size() - 1; ++i) {
      Expression state_prev = state_builder.back();
      vector<Expression> e_t_vector;
      for (int j = 0; j < encoded.size(); ++j) {
        e_t_vector.push_back(concatenate({state_prev, encoded[j],
            position(cg, j, i, encoded.size())}));
      }
      Expression e_t = concatenate_cols(e_t_vector);
      Expression alignment = softmax(tanh(transpose(e_t) * A));
      Expression encoded_matrix = concatenate_cols(encoded);
      Expression context = encoded_matrix * alignment;
      Expression x_t = lookup(cg, tgt_words, tgt_d.Convert(target.words[i]));
      state_builder.add_input(concatenate({context, x_t}));
      Expression y_t = state_builder.back();
      Expression r_t = bias + (R * y_t);
      Expression err = pickneglogsoftmax(r_t, tgt_d.Convert(target.words[i+1]));
      errs.push_back(err);
    }
    return sum(errs);
  }

  vector<unsigned> predict(ComputationGraph& cg, vector<Expression> encoded) {
    state_builder.new_graph(cg);
    state_builder.start_new_sequence();
    Expression decode_start = parameter(cg, p_decode_start);
    Expression A = parameter(cg, p_A);
    Expression R = parameter(cg, p_R);
    Expression bias = parameter(cg, p_bias);

    state_builder.add_input(decode_start);
    vector<unsigned> output;
    unsigned cw = tgt_d.Convert("<s>");
    output.push_back(cw);
    while (cw != tgt_d.Convert("</s>") && output.size() < 50) {
      Expression state_prev = state_builder.back();
      vector<Expression> e_t_vector;
      for (int j = 0; j < encoded.size(); ++j) {
        e_t_vector.push_back(concatenate({state_prev, encoded[j],
            position(cg, j, output.size(), encoded.size())}));
      }
      Expression e_t = concatenate_cols(e_t_vector);
      Expression alignment = softmax(tanh(transpose(e_t) * A));
      Expression encoded_matrix = concatenate_cols(encoded);
      Expression context = encoded_matrix * alignment;
      Expression x_t = lookup(cg, tgt_words, cw);
      state_builder.add_input(concatenate({context, x_t}));
      Expression y_t = state_builder.back();
      Expression r_t = bias + (R * y_t);
      Expression prob = softmax(r_t);
      vector<float> probs = as_vector(cg.incremental_forward());

      float best_prob = probs[0];
      unsigned best_word = 0;
      for (int k = 1; k < probs.size(); ++k) {
        if (probs[k] > best_prob) {
          best_prob = probs[k];
          best_word = k;
        }
      }

      cw = best_word;
      output.push_back(cw);
    }
    return output;
  }
};

Expression BuildComputationGraph(Instance instance,
  ComputationGraph& cg,
  Encoder<LSTMBuilder> encoder,
  Decoder<LSTMBuilder> decoder) {

  Sentence source = instance.source;
  Sentence target = instance.target;

  vector<Expression> encoded = encoder.encode_sentence(cg, source);
  return decoder.decode_train(cg, target, encoded);
}

vector<unsigned> Translate(Sentence source,
  ComputationGraph& cg,
  Encoder<LSTMBuilder> encoder,
  Decoder<LSTMBuilder> decoder) {
  vector<Expression> encoded = encoder.encode_sentence(cg, source);
  return decoder.predict(cg, encoded);
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  string train_src_file = "../data/train.src";
  string train_tgt_file = "../data/train.tgt";
  string dev_src_file = "../data/dev.src";
  string dev_tgt_file = "../data/dev.tgt";
  string test_src_file = "../data/test.src";

  string lmodel = "in_model";
  string smodel = "out_model";

  Model model;

  vector<Sentence> train_src = readData(train_src_file);
  vector<Sentence> train_tgt = readData(train_tgt_file);
  vector<Sentence> dev_src = readData(dev_src_file);
  vector<Sentence> dev_tgt = readData(dev_tgt_file);
  vector<Sentence> test_src = readData(test_src_file);

  buildDict(train_src, true);
  buildDict(train_tgt, false);

  src_d.Freeze();
  tgt_d.Freeze();

  src_d.SetUnk("<unk>");
  tgt_d.SetUnk("<unk>");

  SRC_VOCAB_SIZE = src_d.size();
  TGT_VOCAB_SIZE = tgt_d.size();

  Encoder<LSTMBuilder> encoder(model);
  Decoder<LSTMBuilder> decoder(model);

  if (LOAD_MODEL) {
    string fname = lmodel;
    cerr << "Reading parameters from " << fname << "...\n";
    ifstream in(fname);
    assert(in);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  if (TRAIN) {
    Trainer* sgd = nullptr;
    sgd = new SimpleSGDTrainer(&model);
    vector<Instance> training_instances = buildInstances(train_src, train_tgt);
    vector<Instance> dev_instances = buildInstances(dev_src, dev_tgt);
    vector<unsigned> order(training_instances.size());
    for (int i = 0; i < order.size(); ++i) order[i] = i;

    unsigned report = 0;
    unsigned report_every_i = 500;
    unsigned dev_every_i_reports = 5;
    unsigned si = training_instances.size();
    unsigned lines = 0;
    int epoch_count = 0;
    bool first = true; 
    double best_loss = 9e99;
    while (true) {
      double loss = 0;
      double num_instances = 0;
      for (int i = 0; i < report_every_i; ++i) {
        if (si == order.size()) {
          si = 0;
          if (first) { first = false; } else { sgd->update_epoch(); epoch_count++;}
          // shuffle training instances
          shuffle(order.begin(), order.end(), *rndeng);
        }

        ComputationGraph cg;
        Instance training_instance = training_instances[order[si]];
        ++si;
       
        Expression err = BuildComputationGraph(training_instance, cg,
          encoder, decoder);
        loss += as_scalar(cg.incremental_forward());
        cg.backward();
        sgd->update();
        ++num_instances;
        ++lines;
      }

      sgd->status();
      cerr << " E = " << (loss / num_instances) 
           << " ppl=" << exp(loss / num_instances) << '\n';
      ++report;

      if (report % dev_every_i_reports == 0) {
        double dloss = 0;
        unsigned dnum_instances = 0;
        for (int i = 0; i < dev_instances.size(); ++i) {
          ComputationGraph dcg;
          Instance dev_instance = dev_instances[i];

          Expression err = BuildComputationGraph(dev_instance, dcg,
            encoder, decoder);
          dloss += as_scalar(dcg.incremental_forward());
          ++dnum_instances;
        }
        double average_d_loss = dloss/dnum_instances;
        cerr << "\n***DEV [epoch=" << (lines / (double)training_instances.size()) 
             << "] E = " << (dloss / dnum_instances) 
             << " ppl=" << exp(dloss / dnum_instances) << "\n";

        if (average_d_loss < best_loss) {
          best_loss = average_d_loss;
          ofstream out(smodel);
          boost::archive::text_oarchive oa(out);
          oa << model;
        }        
      }
    }
  }

  for (int i = 0; i < test_src.size(); ++i) {
    ComputationGraph cg;
    vector<unsigned> output = Translate(test_src[i], cg, encoder, decoder);
    for (int j = 0; j < output.size(); ++j) {
      cout << tgt_d.Convert(output[j]) << " ";
    }
    cout << "\n";
  } 
}
