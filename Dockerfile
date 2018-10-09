FROM dsksd/pytorch:0.4
# Install requirements
RUN pip3 install nltk numpy torch==0.4.1 h5py allennlp

# For docker
ENV DATA_DIR=/static/
# DATA_DIR=$HOME/data/  # For local
RUN mkdir $DATA_DIR

# Download GloVe
ENV GLOVE_DIR=$DATA_DIR/glove
RUN mkdir $GLOVE_DIR
RUN wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
RUN unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR

# Download GloVe for SQuAD
ENV GLOVE_DIR=$DATA_DIR/glove_squad
RUN mkdir $GLOVE_DIR
RUN wget https://github.com/seominjoon/squad/releases/download/glove_squad/glove_squad.6B.zip -O $GLOVE_DIR/glove.6B.zip
RUN unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR

# Download ELMo
ENV ELMO_DIR=$DATA_DIR/elmo
RUN mkdir $ELMO_DIR
RUN wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json -O $ELMO_DIR/options.json
RUN wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 -O $ELMO_DIR/weights.hdf5
