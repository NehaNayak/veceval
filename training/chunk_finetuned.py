import pickle 
import random
import sys

import veceval as ve
import numpy as np
np.random.seed(ve.SEED)

from trainer import Trainer
from index_datasets import IndexWindowCapsDataset

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Reshape

class ChunkFinetunedTrainer(Trainer):
  def __init__(self, config_path, name):
    # Define constants and paths
    self.TASK = ve.CHUNK
    self.MODE = ve.FINE
    self.name = name
    (self.train_data_path, self.checkpoint_path,
     self.embedding_path) = ve.make_paths(self.TASK, self.MODE, self.name)
    
    # Get embeddings
    self.embeddings = pickle.load(open(self.embedding_path, 'r'))
    self.ds = IndexWindowCapsDataset(self.train_data_path, self.embeddings,
                                     has_validation=False, is_testing=ve.IS_TESTING)

    # Define model 
    self.hp = ve.read_hp(config_path)
    self.hp.stop_epochs = ve.STOP_EPOCHS
    self.model = self.build_model()

  def build_model(self):
    model = Sequential()
    model.add(Embedding(input_dim=len(self.ds.vocab),
                        output_dim=ve.EMBEDDING_SIZE,
                        weights=[self.ds.weights],
                        input_length=ve.WINDOW_SIZE))
    model.add(Reshape((ve.EMBEDDING_SIZE * ve.WINDOW_SIZE,)))
    model.add(Dense(output_dim=ve.HIDDEN_SIZE))
    model.add(Activation(ve.TANH))
    model.add(Dropout(ve.DROPOUT_PROB))
    model.add(Dense(input_dim=ve.HIDDEN_SIZE, output_dim=ve.CHUNK_CLASSES))
    model.add(Activation(ve.SOFTMAX))
    ve.compile_other_model(model, self.hp.optimizer)
    return model


def main():
  config_path, name = sys.argv[1:3]
  trainer = ChunkFinetunedTrainer(config_path, name)
  trainer.train_and_test()


if __name__ == "__main__":
  main()
