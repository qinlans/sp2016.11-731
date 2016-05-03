Strategies:
We implemented a global attentional model that took into account the positions of the current target word, the source words, and the length
of the source sentence when calculating the multiplier for the source encodings.  The previous hidden state of the decoder, the encoded source
word, and this position factor were concatenated to form a matrix, which was then multiplied with an alignment matrix and passed through a
non-linearity to generate a weight vector over the source encodings.  Each source encoding was multiplied by its weight and the resulting
vectors were added to give us a context vector.  The context vector was concatenated with the previously decoded word, which was then fed
to the decoder's hidden state RNN.

We also used a bidirectional LSTM for the encoder.

Building:
This program is built similarly to cnn and requires Eigen.

In `src`, you need to first use [`cmake`](http://www.cmake.org/) to generate the makefiles

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen

Then to compile, run

    make -j 2

