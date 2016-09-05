import pickle 
import sys

import veceval as ve
import numpy as np
np.random.seed(ve.SEED)

from trainer import Trainer
from index_datasets import IndexWindowCapsDataset

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Reshape, Merge
from keras.regularizers import l2

class NERFinetunedTrainer(Trainer):
  def __init__(self, config_path, name):
    # Define constants and paths
    self.TASK = ve.NER
    self.MODE = ve.FINE
    self.name = name
    (self.train_data_path, self.checkpoint_path,
     self.embedding_path) = ve.make_paths(self.TASK, self.MODE, self.name)
    
    # Get embeddings
    self.embeddings = pickle.load(open(self.embedding_path, 'r'))
    self.ds = IndexWindowCapsDataset(
      self.train_data_path, self.embeddings, has_validation=True,
      has_caps=True, is_testing=ve.IS_TESTING)

    # Define model 
    self.hp = ve.read_hp(config_path)
    self.hp.stop_epochs = ve.STOP_EPOCHS
    self.model = self.build_model()

  def build_model(self):

    vector_input = Sequential()
    vector_input.add(Embedding(input_dim=len(self.ds.vocab),
                        output_dim=ve.EMBEDDING_SIZE,
                        weights=[self.ds.weights],
                        input_length=ve.WINDOW_SIZE))
    caps_input = Sequential()
    caps_input.add(Embedding(input_dim=ve.CAPS_DIMS,
                        output_dim=ve.CAPS_DIMS,
                        weights=[np.eye(ve.CAPS_DIMS)],
                        input_length=ve.WINDOW_SIZE))
    model = Sequential()
    model.add(Merge([vector_input, caps_input], mode=ve.CONCAT))
    model.add(
        Reshape(((ve.EMBEDDING_SIZE + ve.CAPS_DIMS) * ve.WINDOW_SIZE,)))
    model.add(Dense(output_dim=ve.HIDDEN_SIZE))
    model.add(Activation(ve.TANH))
    model.add(Dropout(ve.DROPOUT_PROB))
    model.add(Dense(input_dim=ve.HIDDEN_SIZE, output_dim=ve.NER_CLASSES))
    model.add(Activation(ve.SOFTMAX))
    ve.compile_other_model(model, self.hp.optimizer)

    return model


  def train(self):
    callbacks = ve.callbacks(self.checkpoint_path, self.hp.stop_epochs)
    history = self.model.fit(
        list(self.ds.X_train), self.ds.Y_train, batch_size=ve.BATCH_SIZE,
        nb_epoch=ve.MAX_EPOCHS, verbose=1,
        validation_data=(list(self.ds.X_val), self.ds.Y_val),
        callbacks=callbacks)


  def evaluate(self, set_to_evaluate=ve.VAL):

    if set_to_evaluate == ve.VAL:
      (X, Y) = self.ds.X_val, self.ds.Y_val
    elif set_to_evaluate == ve.TRAIN:
      (X, Y) = self.ds.X_train, self.ds.Y_train
    else:
      assert set_to_evaluate == ve.TEST and ve.IS_TESTING == True
      (X, Y) = self.ds.X_test, self.ds.Y_test

    predictions = self.model.predict(list(X))
    result = ve.calculate_f1(predictions, Y)

    return set_to_evaluate, result
      

def main():
  config_path, name = sys.argv[1:3]
  trainer = NERFinetunedTrainer(config_path, name)
  trainer.train_and_test()


if __name__ == "__main__":
  main()
