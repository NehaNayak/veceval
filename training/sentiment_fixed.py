import pickle 
import random
import sys

import veceval as ve
import numpy as np
np.random.seed(ve.SEED)

from trainer import Trainer
from embedding_datasets import EmbeddingDataset

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.recurrent import LSTM
from keras.regularizers import l2

class SentimentFixedTrainer(Trainer):
  def __init__(self, config_path, name):
    # Define constants and paths
    self.TASK = ve.SENTIMENT
    self.MODE = ve.FIXED
    self.name = name
    (self.train_data_path, self.checkpoint_path,
     self.embedding_path) = ve.make_paths(self.TASK, self.MODE, self.name)
    
    # Get embeddings
    self.embeddings = pickle.load(open(self.embedding_path, 'r'))
    self.ds = EmbeddingDataset(self.train_data_path, self.embeddings,
                           ve.SENTIMENT_MAX_LEN, has_validation=True,
                           is_testing=False)

    # Define model 
    self.hp = ve.read_hp(config_path)
    self.hp.stop_epochs = ve.STOP_EPOCHS
    self.model = self.build_model()

  def build_model(self):
    model = Sequential()
    model.add(
        LSTM(input_shape=self.ds.X_train.shape[1:], output_dim=ve.HIDDEN_SIZE))
    model.add(Dropout(ve.DROPOUT_PROB))
    model.add(Dense(input_dim=ve.HIDDEN_SIZE,
                    output_dim=ve.SENTIMENT_CLASSES,
                    W_regularizer=l2(self.hp.dense_l2)))
    model.add(Activation(ve.SIGMOID))
    ve.compile_binary_model(model, self.hp.optimizer)
    return model


def main():
  config_path, name = sys.argv[1:3]
  trainer = SentimentFixedTrainer(config_path, name)
  trainer.train_and_test()

if __name__ == "__main__":
  main()
