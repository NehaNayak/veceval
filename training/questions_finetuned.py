import pickle 
import sys

import veceval as ve
import numpy as np
np.random.seed(ve.SEED)

from trainer import Trainer
from index_datasets import IndexDataset

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.regularizers import l2

class QuestionsFinetunedTrainer(Trainer):
  def __init__(self, config_path, name):
    # Define constants and paths
    self.TASK = ve.QUESTIONS
    self.MODE = ve.FINE
    self.name = name
    (self.train_data_path, self.checkpoint_path,
     self.embedding_path) = ve.make_paths(self.TASK, self.MODE, self.name)
    
    # Get embeddings
    self.embeddings = pickle.load(open(self.embedding_path, 'r'))
    self.ds = IndexDataset(self.train_data_path, self.embeddings,
                           ve.QUESTIONS_MAX_LEN, has_validation=False,
                           is_testing=False)

    # Define model 
    self.hp = ve.read_hp(config_path)
    self.hp.stop_epochs = ve.STOP_EPOCHS
    self.model = self.build_model()

  def build_model(self):
    model = Sequential()
    model.add(Embedding(input_dim=len(self.ds.vocab),
                        output_dim=ve.EMBEDDING_SIZE,
                        weights=[self.ds.weights],
                        input_length=ve.QUESTIONS_MAX_LEN,
                        W_regularizer=l2(self.hp.embedding_l2)))
    model.add(LSTM(output_dim=ve.HIDDEN_SIZE))
    model.add(Dropout(ve.DROPOUT_PROB))
    model.add(Dense(input_dim=ve.HIDDEN_SIZE,
                    output_dim=ve.QUESTIONS_CLASSES,
                    W_regularizer=l2(self.hp.dense_l2)))
    model.add(Activation(ve.SOFTMAX))
    ve.compile_other_model(model, self.hp.optimizer)

    return model


def main():
  config_path, name = sys.argv[1:3]
  trainer = QuestionsFinetunedTrainer(config_path, name)
  trainer.train_and_test()

if __name__ == "__main__":
  main()
