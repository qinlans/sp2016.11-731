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
#include <map>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

unsigned GLOVE_DIM = 50;
unsigned SENTENCE_DIM = 100;
unsigned SEM_DIM = 100;
unsigned INPUT_DIM = SEM_DIM + SENTENCE_DIM;
unsigned HIDDEN_DIM = 50;
unsigned PAIRWISE_DIM = 10;
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
    int line;
};


unordered_map<string, vector<float>> getAllWords(string hyp1, string hyp2, string ref, string gloveFile){
  string word; //current word read in
  string line;
  string element;
  unordered_set<string>  words; //a list (basically) of all of the words in the hypothesis and reference-make this a hash?
  unordered_map<string, vector<float>> word2gl; //a map storing all of the strings and their embeddings
  for (string file : {hyp1, hyp2, ref}) {
    //for each word in the input, get the word and put it in the map
    ifstream in(file);
    {

      while(getline(in, line)){
        istringstream iss(line);
        while(getline(iss, word, ' ')) {
            words.insert(word);
        }
      }
    }
  }

  ifstream in(gloveFile);
  {
    while(getline(in, line)){
      //for each word that has a gloVe vector
      //get the word
      istringstream iss(line);
      getline(iss, word, ' ');
      vector<float> gloVe;

      //then if the word is in the map, read in all of the floats that make up the vector, add them to gloVe
      //and add the gloVe ctor to the map (couldn't resist, sorry >_>)
      if(words.find(word) != words.end()){

        while(getline(iss, element, ' ')) {
          gloVe.push_back(stof(element));
        }
        word2gl[word] = gloVe;
      }
    }
  }

  return word2gl;
}


vector<Instance> setVector(string hyp1, string hyp2, string ref, unordered_map<string, vector<float>> word2gl) {
  vector<Instance> instances;
  string line;
  string word;

  //hyp1
  {
    //for each word in the input, get the word and add the gloVe ector to the sentence, add sentence to the instance
    ifstream in(hyp1);
    {
      while(getline(in, line)){
        Sentence sentence;
        Instance instance;
        vector<vector<float>> gloVes;
        istringstream iss(line);
        //this will add the semantic expression/vec/whatever
        while(getline(iss, word, ' ')) {
          gloVes.push_back(word2gl[word]);
        }
        sentence.sem = gloVes;
        instance.hyp1 = sentence;
        instances.push_back(instance);
      }
    }
  }


  //hyp2
  {
    //for each word in the input, get the word and add the gloVe ector to the sentence, add sentence to the instance
    ifstream in(hyp2);
    {
      int counter = 0;
      while(getline(in, line)){
        Sentence sentence;
        vector<vector<float>> gloVes;
        istringstream iss(line);
        //this will add the semantic expression/vec/whatever
        while(getline(iss, word, ' ')) {
          gloVes.push_back(word2gl[word]);
        }
        sentence.sem = gloVes;
        instances[counter].hyp2 = sentence;
        counter++;
      }
    }
  }

  //reference
  {
    //for each word in the input, get the word and add the gloVe ector to the sentence, add sentence to the instance
    ifstream in(ref);
    {
      int counter = 0;
      while(getline(in, line)){
        Sentence sentence;
        vector<vector<float>> gloVes;
        istringstream iss(line);
        //this will add the semantic expression/vec/whatever
        while(getline(iss, word, ' ')) {
          gloVes.push_back(word2gl[word]);
        }
        sentence.sem = gloVes;
        instances[counter].ref = sentence;
        counter++;
      }
    }
  }
  return instances;

}

//this sets the bleu score, meteor score, and correct answer
vector<Instance> setBMC(string bleu_file, string meteor_file, string correct, vector<Instance> instances){
  string line;
  string word;

  //BLEU (for french LOSERS) for hyp1-ref, hyp2-ref, hyp1-hyp2 (to be stored in ref)
  {
    //for each word in the input, get the word and add the gloVe ector to the sentence, add sentence to the instance
    ifstream in(bleu_file);
    {
      int counter = 0;
      while(getline(in, line)){
        float BLEU;
        istringstream iss(line);
        //this will add the semantic expression/vec/whatever
        getline(iss, word, '\t');
        BLEU = stof(word);
        instances[counter].hyp1.BLEU = BLEU;

        getline(iss, word, '\t');
        BLEU = stof(word);
        instances[counter].hyp2.BLEU = BLEU;

        getline(iss, word, '\t');
        BLEU = stof(word);
        instances[counter].ref.BLEU = BLEU;

        counter++;
      }
    }
  }

  //Meteor (for AMERICAN WINNERS) for hyp1-ref, hyp2-ref, hyp1-hyp2 (to be stored in ref)
  {
    //for each word in the input, get the word and add the gloVe ector to the sentence, add sentence to the instance
    ifstream in(meteor_file);
    {
      int counter = 0;
      while(getline(in, line)){
        float meteor;
        istringstream iss(line);
        //this will add the semantic expression/vec/whatever
        getline(iss, word, '\t');
        meteor = stof(word);
        instances[counter].hyp1.meteor = meteor;

        getline(iss, word, '\t');
        meteor = stof(word);
        instances[counter].hyp2.meteor = meteor;

        getline(iss, word, '\t');
        meteor = stof(word);
        instances[counter].ref.meteor = meteor;

        counter++;
      }
    }
  }

  //correct answer
  {
    ifstream in(correct);
    {
      int counter = 0;
      while(getline(in, line)){
        vector<vector<float>> gloVes;
        istringstream iss(line);
        //this will add the semantic expression/vec/whatever
        getline(iss, word, '\n');
        instances[counter].correct = stoi(word);
        instances[counter].line = counter;
        counter++;
      }
    }
  }
  return instances;
}

vector<Instance> setSyn(string shyp1, string shyp2, string sref, vector<Instance> instances) {
  string line;
  string word;
  string element;

  //hypothesis one sentence vect
  {
    ifstream in(shyp1);
    int counter = 0;
    while(getline(in, line)){
      istringstream iss(line);

      vector<float> sentence_vector;

      while(getline(iss, element, ',')) {
        sentence_vector.push_back(stof(element));
      }
      instances[counter].hyp1.syn = sentence_vector;
      counter++;
    }
  }


  //hypothesis two sentence vect
  {
    ifstream in(shyp2);
    int counter = 0;

    while(getline(in, line)){
      istringstream iss(line);

      vector<float> sentence_vector;

      while(getline(iss, element, ',')) {
        sentence_vector.push_back(stof(element));
      }
      instances[counter].hyp2.syn = sentence_vector;
      counter++;
    }
  }
  
  //ref sentence vect
  {
    ifstream in(sref);
    int counter = 0;

    while(getline(in, line)){
      istringstream iss(line);
      vector<float> sentence_vector;

        while(getline(iss, element, ',')) {
          sentence_vector.push_back(stof(element));
        }
        instances[counter].ref.syn = sentence_vector;
        counter++;
    }
  }
  return instances;
}

struct EvaluationGraph {
  Parameters* se;
  Parameters* se_b;
  Parameters* ie;
  Parameters* ie_b;
  Parameters* W_12;
  Parameters* b_12;
  Parameters* W_1r;
  Parameters* b_1r;
  Parameters* W_2r;
  Parameters* b_2r;
  Parameters* V_;
  Parameters* b_;

  explicit EvaluationGraph(Model* m) :
    se(m->add_parameters({SEM_DIM, GLOVE_DIM * GLOVE_DIM})),
    se_b(m->add_parameters({SEM_DIM})),
    ie(m->add_parameters({HIDDEN_DIM, INPUT_DIM})),
    ie_b(m->add_parameters({HIDDEN_DIM})),
    W_12(m->add_parameters({PAIRWISE_DIM, HIDDEN_DIM * 2})),
    b_12(m->add_parameters({PAIRWISE_DIM})),
    W_1r(m->add_parameters({PAIRWISE_DIM, HIDDEN_DIM * 2})),
    b_1r(m->add_parameters({PAIRWISE_DIM})),
    W_2r(m->add_parameters({PAIRWISE_DIM, HIDDEN_DIM * 2})),
    b_2r(m->add_parameters({PAIRWISE_DIM})),
    V_(m->add_parameters({1, PAIRWISE_DIM * 2 + 2})),
    b_(m->add_parameters({1}))
    {

    }


  Expression buildComputationGraph(Instance instance,
   ComputationGraph& cg, Model* m) {
    Expression sem_embed = parameter(cg, se);
    Expression sem_bias = parameter(cg, se_b);
    Expression input_embed = parameter(cg, ie);
    Expression input_bias = parameter(cg, ie_b);
    Expression W12 = parameter(cg, W_12);
    Expression b12 = parameter(cg, b_12);
    Expression W1r = parameter(cg, W_1r);
    Expression b1r = parameter(cg, b_1r);
    Expression W2r = parameter(cg, W_2r);
    Expression b2r = parameter(cg, b_2r);
    Expression V = parameter(cg, V_);
    Expression b = parameter(cg, b_);

    // Create embedding from syntax and semantic vector
    // {SENTENCE_DIM, 1} result
    Expression hyp1syn = input(cg, {SENTENCE_DIM, 1}, instance.hyp1.syn);
    Expression hyp2syn = input(cg, {SENTENCE_DIM, 1}, instance.hyp2.syn);
    Expression refsyn = input(cg, {SENTENCE_DIM, 1}, instance.ref.syn);

    vector<Expression> hyp1sem_vectors;
    vector<Expression> hyp2sem_vectors;
    vector<Expression> refsem_vectors;
    for (int i = 0; i < instance.hyp1.sem.size(); ++i) {
      if (instance.hyp1.sem[i].size() != 0) {
        hyp1sem_vectors.push_back(input(cg, {GLOVE_DIM, 1},
            instance.hyp1.sem[i]));
      }
    }
    for (int i = 0; i < instance.hyp2.sem.size(); ++i) {
      if (instance.hyp2.sem[i].size() != 0) {
        hyp2sem_vectors.push_back(input(cg, {GLOVE_DIM, 1},
            instance.hyp2.sem[i]));
      }
    }
    for (int i = 0; i < instance.ref.sem.size(); ++i) {
      if (instance.ref.sem[i].size() != 0) {
        refsem_vectors.push_back(input(cg, {GLOVE_DIM, 1},
            instance.ref.sem[i]));
      }
    }
    Expression hyp1sem_matrix = concatenate_cols(hyp1sem_vectors);
    Expression hyp2sem_matrix = concatenate_cols(hyp2sem_vectors);
    Expression refsem_matrix = concatenate_cols(refsem_vectors);

    // {SEM_DIM, 1} result
    Expression hyp1sem = tanh(sem_embed * reshape(hyp1sem_matrix *
        transpose(hyp1sem_matrix), {GLOVE_DIM * GLOVE_DIM, 1}));
    Expression hyp2sem = tanh(sem_embed * reshape(hyp2sem_matrix *
        transpose(hyp2sem_matrix), {GLOVE_DIM * GLOVE_DIM, 1}));
    Expression refsem = tanh(sem_embed * reshape(refsem_matrix *
        transpose(refsem_matrix), {GLOVE_DIM * GLOVE_DIM, 1}));

    // {HIDDEN_DIM, 1} result
    Expression x1 = tanh(input_embed * concatenate({hyp1syn, hyp1sem}));
    Expression x2 = tanh(input_embed * concatenate({hyp2syn, hyp2sem}));
    Expression xref = tanh(input_embed * concatenate({refsyn, refsem}));

    // Create pairwise vectors
    // {PAIRWISE_DIM, 1} result
    Expression h12 = tanh(W12 * concatenate({x1, x2}) + b12);
    Expression h1r = tanh(W1r * concatenate({x1, xref}) + b1r);
    Expression h2r = tanh(W2r * concatenate({x2, xref}) + b2r);

    // Combination of evaluation input
    // {PAIRWISE_DIM * 3 + 4, 1} result
    Expression BLEU1 = input(cg, instance.hyp1.BLEU);
    Expression BLEU2 = input(cg, instance.hyp2.BLEU);
    Expression meteor1 = input(cg, instance.hyp1.meteor);
    Expression meteor2 = input(cg, instance.hyp2.meteor);
    Expression hyp1 = concatenate({h1r, h12, BLEU1, meteor1});
    Expression hyp2 = concatenate({h2r, h12, BLEU2, meteor2});
    Expression u = concatenate({V * hyp1 + b, V * hyp2 + b });

    return u;
  }


};

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  string hyp1 = "../data/hyp1lower.txt";
  string hyp2 = "../data/hyp2lower.txt";
  string ref = "../data/reflower.txt";
  string gloveFile = "../data/glove.6B.50d.txt";
  string bleu = "../data/bleu.txt";
  string meteor = "../data/meteor.txt";
  string correct = "../data/train.gold";
  string shyp1 = "../data/hyp1vectors.txt";
  string shyp2 = "../data/hyp2vectors.txt";
  string sref = "../data/refvectors.txt";

  bool load_model = false;
  string lmodel = "in_model";
  string smodel = "model";

  Model model;

  if (load_model) {
    string fname = lmodel;
    cerr << "Reading parameters from " << fname << "...\n";
    ifstream in(fname);
    assert(in);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  Trainer* sgd = nullptr;
  sgd = new SimpleSGDTrainer(&model);
  EvaluationGraph evaluator(&model);

  unordered_map<string, vector<float>> word2gl =
      getAllWords(hyp1, hyp2, ref, gloveFile);
  vector<Instance> instances = setVector(hyp1, hyp2, ref, word2gl);
  instances = setBMC(bleu, meteor, correct, instances);
  instances = setSyn(shyp1, shyp2, sref, instances);

  // Shuffles instances with gold data
  vector<unsigned> order(26208);
  for (int i = 0; i < order.size(); ++i) order[i] = i;
  shuffle(order.begin(), order.end(), *rndeng);

  // Picks subsets of the gold data to be training and dev sets
  vector<unsigned> training(order.begin(), order.end() - 5000);
  vector<unsigned> dev(order.end() - 5000, order.end());  

  vector<unsigned> training_order(training.size());
  for (int i = 0; i < training_order.size(); ++i) training_order[i] = i;

  bool first = true;
  unsigned report = 0;
  unsigned report_every_i = 20;
  unsigned dev_every_i_reports = 20;
  unsigned lines = 0;
  unsigned si = training_order.size();
  double best = 0;
  while(1) {
    double loss = 0;
    unsigned num_instances = 0;
    unsigned num_correct = 0;
    for (int i = 0; i < report_every_i; ++i) {
      if (si == training_order.size()) {
        si = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        // shuffle training instances
        shuffle(training_order.begin(), training_order.end(), *rndeng);
      }

      ComputationGraph cg;
      Instance training_instance = instances[order[training_order[si]]];
      ++si;

      Expression u = evaluator.buildComputationGraph(training_instance, cg,
          &model);   
      vector<float> hyp_probs = as_vector(cg.incremental_forward());
      
      int pred = 0;
      if (hyp_probs[0] > hyp_probs[1]) { pred = -1; }
      else { pred = 1; }
  
      int correct_hyp = training_instance.correct;
      if (correct_hyp == pred) {
        ++num_correct;
      }

      int update_pred = 0;
      if (correct_hyp == -1) { update_pred = 0; }
      else { update_pred = 1; }

      Expression loss_exp = pickneglogsoftmax(u, update_pred);
      loss += as_scalar(cg.incremental_forward());
      cg.backward();
      sgd->update();
      ++num_instances;
      ++lines;
    }

    //sgd->status();
    //cerr << " E = " << (loss / num_instances)
    //     << " ppl = " << exp(loss / num_instances) 
    //     << " accuracy = " << (float(num_correct) / num_instances) << "\n";
    report++;

    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      unsigned dnum_instances = 0;
      unsigned dnum_correct = 0;
      for (int i = 0; i < dev.size(); ++i) {
        ComputationGraph dcg;
        Instance dev_instance = instances[dev[i]];
        Expression du = evaluator.buildComputationGraph(dev_instance, dcg,
            &model);
        vector<float> dhyp_probs = as_vector(dcg.incremental_forward());

        int dpred = 0;
        if (dhyp_probs[0] > dhyp_probs[1]) { dpred = -1; }
        else { dpred = 1; }

        int dcorrect_hyp = dev_instance.correct;
        if (dcorrect_hyp == dpred) {
          ++dnum_correct;
        }

        int dupdate_pred = 0;
        if (dcorrect_hyp == -1) { dupdate_pred = 0; }
        else { dupdate_pred = 1; }

        Expression dloss_exp = pickneglogsoftmax(du, dupdate_pred);
        dloss += as_scalar(dcg.incremental_forward());
        ++dnum_instances;
      }

      cerr << "\n***DEV [epoch=" << (lines / (double)training.size())
           << "] E = " << (dloss / dnum_instances)
           << " ppl= " << exp(dloss / dnum_instances)
           << " accuracy = " << (float(dnum_correct) / dnum_instances) << "\n";

      float daccuracy = float(dnum_correct) / dnum_instances;
      if (daccuracy > best) {
        best = daccuracy;
        ofstream out(smodel);
        boost::archive::text_oarchive oa(out);
        oa << model;
      }
      cerr << " best accuracy = " << best << "\n";
    }
  }

}
