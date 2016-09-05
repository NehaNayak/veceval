from veceval import *

 
def convert_sequences_to_dataset(sequences, max_len, vocab_dict):
  """Create a np.array of indices given sequences and a vocab dictionary."""
  dataset = np.zeros((len(sequences), max_len))

  for (i, sequence) in enumerate(sequences): 
    for (j, word) in enumerate(sequence[:max_len]):
      dataset[i][j] = lookup(word, vocab_dict)
    for j in range(len(sequence), max_len):
      dataset[i][j] = lookup(PAD, vocab_dict)

  return dataset


def convert_windows_to_dataset(windows, vocab_dict, has_caps):
  """Create a np.array of indices given sequences and a vocab dictionary."""
  dataset = np.zeros((len(windows), WINDOW_SIZE))
  caps_dataset = np.zeros((len(windows), WINDOW_SIZE))

  for (i, window) in enumerate(windows): 
    for (j, word) in enumerate(window):
      dataset[i][j] = lookup(word.lower(), vocab_dict)
      caps_dataset[i][j] = caps_feature(word)
  if has_caps:
    return dataset, caps_dataset
  else:
    return dataset


class IndexWindowCapsDataset(object):
  def __init__(self, train_pickle_path, embeddings, 
               is_testing=False, has_validation=True, has_caps=False):

    (self.vocab, self.vocab_dict,
     self.weights) = make_vocab_and_weights(embeddings)

    self.window_size = WINDOW_SIZE
    self.is_testing = is_testing
    self.has_validation = has_validation 

    datasets = self.load_datasets(train_pickle_path, has_validation, has_caps)
    (self.X_train, self.Y_train,
     self.X_val, self.Y_val,
     self.X_test, self.Y_test) = datasets
 
  def load_datasets(self, train_pickle_path, has_validation, has_caps):
    """Convert one set of examples into keras-friendly data structures.
    """

    if has_validation:
      X_train_sequences, Y_train, _ = read_data_pickle(train_pickle_path)
      val_pickle_path = train_pickle_path.replace("train", "dev")
      X_val_sequences, Y_val, _ = read_data_pickle(val_pickle_path)
    else:
      X_sequences, Y, _ = read_data_pickle(train_pickle_path)
      split_at = int(len(X_sequences) * 0.9)
      X_train_sequences, X_val_sequences = (X_sequences[:split_at],
                                            X_sequences[split_at:])
      Y_train, Y_val = (Y[:split_at],
                        Y[split_at:])

    X_train = convert_windows_to_dataset(X_train_sequences, self.vocab_dict, has_caps)
    X_val = convert_windows_to_dataset(X_val_sequences, self.vocab_dict, has_caps)

    if not self.is_testing:
      return X_train, Y_train, X_val, Y_val, None, None
    else:
      test_pickle_path = train_pickle_path.replace("train", "test")
      X_test_sequences, Y_test, _ = read_data_pickle(test_pickle_path)
      X_test = convert_windows_to_dataset(X_test_sequences, self.vocab_dict, has_caps)
      return X_train, Y_train, X_val, Y_val, X_test, Y_test


class IndexPairDataset(object):
  def __init__(self, train_pickle_path, embeddings, max_len, is_testing=False,
               has_validation=True):

    (self.vocab, self.vocab_dict,
     self.weights) = make_vocab_and_weights(embeddings)

    self.max_len = max_len
    self.is_testing = is_testing
    self.has_validation = has_validation 

    datasets = self.load_datasets(train_pickle_path, embeddings)
    (self.X_p_train, self.X_h_train, self.Y_train,
     self.X_p_val, self.X_h_val, self.Y_val,
     self.X_p_test, self.X_h_test, self.Y_test) = datasets
 
  def load_dataset_from_path(self, pickle_path, split_validation=False):
    """Convert one set of examples into keras-friendly data structures.
    """
    (X_p_sequences, 
     X_h_sequences, Y, _) = read_data_pair_pickle(pickle_path)
    if split_validation:
      assert False
    else:
      X_p = convert_sequences_to_dataset(X_p_sequences, self.max_len, self.vocab_dict)
      X_h = convert_sequences_to_dataset(X_h_sequences, self.max_len, self.vocab_dict)
      return X_p, X_h, Y

  def load_datasets(self, train_pickle_path, embeddings):
    """Load train, dev and sometimes test datasets.
    """

    val_pickle_path = train_pickle_path.replace("train", "dev")

    if self.has_validation:
      X_p_train, X_h_train, Y_train = self.load_dataset_from_path(train_pickle_path)
      X_p_val, X_h_val, Y_val = self.load_dataset_from_path(val_pickle_path)
    else:
      assert False
    if not self.is_testing:
      return (X_p_train, X_h_train, Y_train,
              X_p_val, X_h_val, Y_val, None, None, None)
    else:
      test_pickle_path = train_pickle_path.replace("train", "test")
      X_p_test, X_h_test, Y_test = self.load_dataset_from_path(test_pickle_path)

      return (X_p_train, X_h_train, Y_train,
              X_p_val, X_h_val, Y_val,
              X_p_test, X_h_test, Y_test)
 

class IndexDataset(object):
  def __init__(self, train_pickle_path, embeddings, max_len,
               is_testing=False, has_validation=True):

    (self.vocab, self.vocab_dict,
     self.weights) = make_vocab_and_weights(embeddings)

    self.max_len = max_len
    self.is_testing = is_testing
    self.has_validation = has_validation 

    datasets = self.load_datasets(train_pickle_path, has_validation)
    (self.X_train, self.Y_train,
     self.X_val, self.Y_val,
     self.X_test, self.Y_test) = datasets
 
  def load_datasets(self, train_pickle_path, has_validation=True):
    """Convert one set of examples into keras-friendly data structures.
    """
    if has_validation:
      val_pickle_path = train_pickle_path.replace("train", "dev")
      X_train_sequences, Y_train, _ = read_data_pickle(train_pickle_path)
      X_val_sequences, Y_val, _ = read_data_pickle(val_pickle_path)
      
    else:
      X_sequences, Y, _ = read_data_pickle(train_pickle_path)

      split_at = int(len(X_sequences) * 0.9)
      X_train_sequences, X_val_sequences = (X_sequences[:split_at],
                                            X_sequences[split_at:])
      Y_train, Y_val = (Y[:split_at],
                        Y[split_at:])
    X_train = convert_sequences_to_dataset(X_train_sequences, self.max_len,
                                           self.vocab_dict)
    X_val = convert_sequences_to_dataset(X_val_sequences, self.max_len,
                                           self.vocab_dict)

    if not self.is_testing:
      return X_train, Y_train, X_val, Y_val, _, _

    else:
      test_pickle_path = train_pickle_path.replace("train", "test")
      X_test_sequences, Y_test, _ = read_data_pickle(test_pickle_path)
      X_test = convert_sequences_to_dataset(X_test_sequences, self.max_len,
                                            self.vocab_dict)
      return X_train, Y_train, X_val, Y_val, X_test, Y_test

