#!/usr/bin/env bash
# Install requirements
pip3 install nltk numpy torch==0.4.1 h5py allennlp

DATA_DIR=/static/  # For docker
# DATA_DIR=$HOME/data/  # For local
mkdir $DATA_DIR

# Download GloVe
GLOVE_DIR=$DATA_DIR/glove
mkdir $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR
rm $GLOVE_DIR/glove.6B.zip

# Download ELMo
ELMO_DIR=$DATA_DIR/elmo
mkdir $ELMO_DIR
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json -O $ELMO_DIR/options.json
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 -O $ELMO_DIR/weights.hdf5

