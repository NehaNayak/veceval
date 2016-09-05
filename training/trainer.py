import os
import veceval as ve
import time

class Trainer(object):

  def __init__(self, is_testing=False):
    self.is_testing = is_testing
    pass

  def evaluate(self, set_to_evaluate=ve.VAL): # Overridden for NER, NLI
    if set_to_evaluate == ve.VAL:
      (X, Y) = self.ds.X_val, self.ds.Y_val
    else:
      assert set_to_evaluate == ve.TRAIN
      (X, Y) = self.ds.X_train, self.ds.Y_train
    _, acc = self.model.evaluate(X, Y)
    return set_to_evaluate, acc

  def load_data(self):
    pass

  def build_model(self):
    pass

  def print_result(self, set_to_evaluate=ve.VAL):
    log_file = os.environ["LOG_FILE"]
    affiliation = os.environ["AFFILIATION"]
    set_to_evaluate, result = self.evaluate(set_to_evaluate)
    with open(log_file, 'a') as f:
      f.write("\t".join([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                         affiliation, self.name, self.TASK, self.MODE,
                         set_to_evaluate, str(result)]))
      f.write("\n")

  def train(self): # Overridden for NER, NLI
    callbacks = ve.callbacks(self.checkpoint_path, self.hp.stop_epochs)
    history = self.model.fit(
        self.ds.X_train, self.ds.Y_train, batch_size=ve.BATCH_SIZE,
        nb_epoch=ve.MAX_EPOCHS, verbose=1, validation_data=(self.ds.X_val,
                                                            self.ds.Y_val),
        callbacks=callbacks)

  def test(self):
    self.model.load_weights(self.checkpoint_path)
    self.evaluate()

  def train_and_test(self):
    self.train()
    self.test()
    self.print_result(ve.TRAIN)
    self.print_result(ve.VAL)
