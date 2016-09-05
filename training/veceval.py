import numpy as np
import os
import pickle
import re
import sys
import keras
from keras.optimizers import Adagrad, Adadelta, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import categorical_probas_to_classes
from sklearn.metrics import f1_score

POS = "pos"
SENTIMENT = "sentiment"
QUESTIONS = "questions"
CHUNK = "chunk"
NLI = "nli"
NER = "ner"

TRAIN = "train"
TEST = "test"
VAL = "val"

FINE = "finetuned"
FIXED = "fixed"

# General hyperparameters
DROPOUT_PROB = 0.5
EMBEDDING_SIZE = 50
HIDDEN_SIZE = 50
WINDOW_SIZE = 5
CAPS_DIMS = 5
STOP_EPOCHS = 15
POS_STOP_EPOCHS = 10
MAX_EPOCHS = 500
BATCH_SIZE = 128

# Task-specific hyperparameters
SENTIMENT_MAX_LEN = 50
QUESTIONS_MAX_LEN = 30
NLI_MAX_LEN = 10

QUESTIONS_CLASSES = 6
CHUNK_CLASSES = 23
POS_CLASSES = 12
NLI_CLASSES = 5
NER_CLASSES = 8
SENTIMENT_CLASSES = 2

# String constants for Keras
TANH = "tanh"
SOFTMAX = "softmax"
SIGMOID = "sigmoid"
CONCAT = "concat"


UNK = "UNK"
PAD = "PAD"
SEED = 137


def make_paths(task, mode, name):
  
  train_data_path = "".join([os.environ["ROOTDIR"], "/data/", task, "/train.pickle"])
  checkpoint_path = "".join([os.environ["CHECKPOINT_HOME"], "/",  name, "_", task, "_",
                             mode, ".ckpt"])
  embedding_path = "".join([os.environ["PICKLES_HOME"], "/", name, ".pickle"])
  return train_data_path, checkpoint_path, embedding_path


class Hyperparams():
  def __init__(self, path):
    hp = {}

    for line in open(path,"r"):
      if not line.startswith("#"):
       (key, val) = line.split()
       hp[key] = val
  
    # Layer choices
    self.dense_l2 = float(hp.get("dense_l2", 0)) 
    self.embedding_l2 = float(hp.get("embedding_l2", 0)) 

    # Optimization
    self.optimizer = hp["optimizer"]
    if "learning_rate" in hp:
      if self.optimizer == "Adagrad":
        self.optimizer = Adagrad(lr=float(hp["learning_rate"]))
      elif self.optimizer == "Adadelta":
        self.optimizer = Adadelta(lr=10.0*float(hp["learning_rate"]))
      elif self.optimizer == "RMSprop":
        self.optimizer = RMSprop(lr=float(hp["learning_rate"])/10.0)

    self.stop_epochs = None  # Fill in later


def read_hp(config_path):
  return Hyperparams(config_path)


def callbacks(checkpoint_path, stop_epochs):
  early_stopping = EarlyStopping(monitor="val_acc", 
      patience=stop_epochs, verbose=1)
  model_checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_acc", 
      verbose=1, save_best_only=True)
  return [early_stopping, model_checkpoint]


def make_vocab_and_weights(embeddings):
  """Make a np.array with embedding weights and accompanying vocab DSs.
  """
  assert UNK in embeddings and PAD in embeddings
  vocab = sorted(embeddings.keys())

  vocab_dict = {word: i for i, word in enumerate(vocab)}
  embedding_weights = [embeddings[word] for word in vocab]

  return (vocab, vocab_dict, np.array(embedding_weights))


def read_data_pickle(pickle_path):
  """Read in examples and label map from pickle file.
  """
  dataset, label_map = pickle.load(open(pickle_path,"r"))
  examples = [example for example, _ in dataset]
  labels = [label for _, label in dataset]
  return examples, np.array(labels), label_map


def read_data_pair_pickle(pickle_path):
  """Read in examples and label map from pickle file.
  """
  dataset, label_map = pickle.load(open(pickle_path,"r"))
  examples_p = [example for example, _, _ in dataset]
  examples_h = [example for _, example, _ in dataset]
  labels = [label for _, _, label in dataset]
  return examples_p, examples_h, np.array(labels), label_map


def lookup(key, lookup_dict):
  """Cleaner dictionary lookup."""
  return lookup_dict.get(key, lookup_dict[UNK])


lower = re.compile("^[a-z]+$")
upper = re.compile("^[A-Z]+$")
init_caps = re.compile("^[A-Z][a-z]+$")
one_upper = re.compile("^[a-z]*[A-Z][a-z]*$")


def caps_feature(word):
  if lower.match(word):
    return 0
  elif init_caps.match(word):
    return 1
  elif upper.match(word):
    return 2
  elif one_upper.match(word):
    return 3
  else:
    return 4


def compile_binary_model(model, optimizer):
  """Compile a model with binary crossentropy and given optimizer.
  """
  sys.stderr.write("Beginning to compile the model.\n")
  model.compile(  loss = "binary_crossentropy",
                  optimizer = optimizer,
                  metrics = ["accuracy"],
              )
  sys.stderr.write("Finished compiling the model.\n")


def compile_other_model(model, optimizer):
  """Compile a model with binary crossentropy and given optimizer.
  """
  sys.stderr.write("Beginning to compile the model.\n")
  model.compile(  loss = "categorical_crossentropy",
                  optimizer = optimizer,
                  metrics = ["accuracy"],
              )
  sys.stderr.write("Finished compiling the model.\n")


def calculate_f1(predictions, actual):
  return f1_score(categorical_probas_to_classes(actual),
                  categorical_probas_to_classes(predictions),
                  average="micro")
