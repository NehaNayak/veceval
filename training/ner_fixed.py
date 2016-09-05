import pickle 
import sys

import veceval as ve
import numpy as np
np.random.seed(ve.SEED)

from trainer import Trainer
from embedding_datasets import EmbeddingWindowCapsDataset

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.regularizers import l2

class NERFixedTrainer(Trainer):
  def __init__(self, config_path, name):
    # Define constants and paths
    self.TASK = ve.NER
    self.MODE = ve.FIXED
    self.name = name
    (self.train_data_path, self.checkpoint_path,
     self.embedding_path) = ve.make_paths(self.TASK, self.MODE, self.name)
    
    # Get embeddings
    self.embeddings = pickle.load(open(self.embedding_path, 'r'))
    self.ds = EmbeddingWindowCapsDataset(
        self.train_data_path, self.embeddings, has_validation=True,
        has_caps=True, is_testing=ve.IS_TESTING)

    # Define model 
    self.hp = ve.read_hp(config_path)
    self.hp.stop_epochs = ve.STOP_EPOCHS
    self.model = self.build_model()

  def build_model(self):
    model = Sequential()
    model.add(Dense(input_shape=self.ds.X_train.shape[1:],
                    output_dim=ve.HIDDEN_SIZE,
                    W_regularizer=l2(self.hp.dense_l2)))
    model.add(Activation(ve.TANH))
    model.add(Dropout(ve.DROPOUT_PROB))
    model.add(Dense(input_dim=ve.HIDDEN_SIZE,
                    output_dim=ve.NER_CLASSES,
                    W_regularizer=l2(self.hp.dense_l2)))
    model.add(Activation(ve.SOFTMAX))
    ve.compile_other_model(model, self.hp.optimizer)
    return model

  def evaluate(self, set_to_evaluate=ve.VAL):
    if set_to_evaluate == ve.VAL:
      (X, Y) = self.ds.X_val, self.ds.Y_val
    elif set_to_evaluate == ve.TRAIN:
      (X, Y) = self.ds.X_train, self.ds.Y_train
    else:
      assert set_to_evaluate == ve.TEST and ve.IS_TESTING == True
      (X, Y) = self.ds.X_test, self.ds.Y_test
  
    predictions = self.model.predict(X)
    result = ve.calculate_f1(predictions, Y)

    return set_to_evaluate, result
      

def main():
  config_path, name = sys.argv[1:3]
  trainer = NERFixedTrainer(config_path, name)
  trainer.train_and_test()


if __name__ == "__main__":
  main()
