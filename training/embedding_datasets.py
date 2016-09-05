from veceval import *


def convert_sequences_to_dataset(sequences, max_len, embeddings):
  """Create a np.array of embeddings given sequences and a dictionary."""
  dataset_list = []
  dummy_embedding = lookup(PAD, embeddings)

  for (i, sequence) in enumerate(sequences):
    embedding_list = []
    for word in sequence[:max_len]:
      embedding = lookup(word, embeddings)
      embedding_list.append(embedding)
    for _ in range(len(sequence), max_len):
      embedding_list.append(dummy_embedding)
    dataset_list.append(embedding_list)

  return np.array(dataset_list)


def convert_windows_to_dataset(windows, embeddings, has_caps=False):
  """Create a np.array of embeddings given sequences and a dictionary."""
  dataset_list = []
  dummy_embedding = lookup(PAD, embeddings)

  if has_caps:
    reshape_shape = ((dummy_embedding.shape[0] + 5)* WINDOW_SIZE, )
    eye = np.eye(5)
    for (i, window) in enumerate(windows):
      embedding_list = []
      for word in window:
        embedding = lookup(word.lower(), embeddings)
        embedding_list.append(np.concatenate([embedding,
                                              eye[caps_feature(word)]]))
      dataset_list.append(np.array(embedding_list).reshape(reshape_shape))
    return np.array(dataset_list)
  else:
    reshape_shape = (dummy_embedding.shape[0] * WINDOW_SIZE, )
    for (i, window) in enumerate(windows):
      embedding_list = []
      for word in window:
        embedding = lookup(word.lower(), embeddings)
        embedding_list.append(embedding)
      dataset_list.append(np.array(embedding_list).reshape(reshape_shape))
    return np.array(dataset_list)

class EmbeddingPairDataset(object):
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

  def load_dataset_from_path(self, pickle_path, embeddings, 
                             split_validation=False):
    """Convert one set of examples into keras-friendly data structures.
    """
    (X_p_sequences, 
     X_h_sequences, Y, _) = read_data_pair_pickle(pickle_path)

    X_p = convert_sequences_to_dataset(X_p_sequences, self.max_len, embeddings)
    X_h = convert_sequences_to_dataset(X_h_sequences, self.max_len, embeddings)
    return X_p, X_h, Y

  def load_datasets(self, train_pickle_path, embeddings):
    """Load train, dev and sometimes test datasets.
    """

    if self.has_validation:
    
      val_pickle_path = train_pickle_path.replace("train", "dev")

      X_p_train, X_h_train, Y_train = self.load_dataset_from_path(train_pickle_path,
                                                     embeddings)
      X_p_val, X_h_val, Y_val = self.load_dataset_from_path(val_pickle_path,
                                                 embeddings)
    else:
      assert(False)

    if not self.is_testing:
      return (X_p_train, X_h_train, Y_train, 
              X_p_val, X_h_val, Y_val, None, None, None)
    else:
      test_pickle_path = train_pickle_path.replace("train", "test")
      X_p_test, X_h_test, Y_test = self.load_dataset_from_path(test_pickle_path, embeddings)

      return (X_p_train, X_h_train, Y_train,
              X_p_val, X_h_val, Y_val,
              X_p_test, X_h_test, Y_test)


class EmbeddingDataset(object):
  def __init__(self, train_pickle_path, embeddings, max_len, is_testing=False,
               has_validation=True):

    (self.vocab, self.vocab_dict,
     self.weights) = make_vocab_and_weights(embeddings)

    self.max_len = max_len
    self.is_testing = is_testing
    self.has_validation = has_validation

    datasets = self.load_datasets(train_pickle_path, embeddings, has_validation)
    (self.X_train, self.Y_train,
     self.X_val, self.Y_val,
     self.X_test, self.Y_test) = datasets 


  def load_datasets(self, train_pickle_path, embeddings, 
                             has_validation=True):
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
    X_train = convert_sequences_to_dataset(X_train_sequences, self.max_len, embeddings)
    X_val = convert_sequences_to_dataset(X_val_sequences, self.max_len, embeddings)

    if not self.is_testing:
      return X_train, Y_train, X_val, Y_val, None, None
    else:
      test_pickle_path = train_pickle_path.replace("train", "test")
      X_test_sequences, Y_test, _ = read_data_pickle(test_pickle_path)
      X_test = convert_sequences_to_dataset(X_test_sequences, self.max_len,
                                            embeddings)
      return X_train, Y_train, X_val, Y_val, X_test, Y_test


  def unused_load_datasets(self, train_pickle_path, embeddings):
    """Load train, dev and sometimes test datasets.
    """

    if self.has_validation:
    
      val_pickle_path = train_pickle_path.replace("train", "dev")

      X_train, Y_train = self.load_dataset_from_path(train_pickle_path,
                                                     embeddings)
      X_val, Y_val = self.load_dataset_from_path(val_pickle_path,
                                                 embeddings)
    else:
      (X_train, Y_train,
       X_val, Y_val) = self.load_dataset_from_path(train_pickle_path, 
                                                   embeddings, True)

    if not self.is_testing:
      return (X_train, Y_train, X_val, Y_val, None, None)
    else:
      test_pickle_path = train_pickle_path.replace("train", "test")
      X_test, Y_test = self.load_dataset_from_path(test_pickle_path, embeddings)

      return (X_train, Y_train, X_val, Y_val, X_test, Y_test)


class EmbeddingWindowCapsDataset(object):
  def __init__(self, train_pickle_path, embeddings,
               is_testing=False, has_validation=True, has_caps=True):

    (self.vocab, self.vocab_dict,
     self.weights) = make_vocab_and_weights(embeddings)

    self.window_size = WINDOW_SIZE
    self.is_testing = is_testing
    self.has_validation = has_validation

    datasets = self.load_datasets(train_pickle_path, embeddings, has_validation,
                                  has_caps)
    (self.X_train, self.Y_train,
     self.X_val, self.Y_val,
     self.X_test, self.Y_test) = datasets 

  def load_datasets(self, train_pickle_path, embeddings, 
                             has_validation, has_caps):
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

    X_train = convert_windows_to_dataset(X_train_sequences, embeddings, has_caps)
    X_val = convert_windows_to_dataset(X_val_sequences, embeddings, has_caps)

    if not self.is_testing:
      return X_train, Y_train, X_val, Y_val, None, None
    else: 
      test_pickle_path = train_pickle_path.replace("train", "test")
      X_test_sequences, Y_test, _ = read_data_pickle(test_pickle_path)
      X_test = convert_windows_to_dataset(X_test_sequences, embeddings, has_caps)
      return X_train, Y_train, X_val, Y_val, X_test, Y_test

