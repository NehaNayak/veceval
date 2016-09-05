import sys
import pickle
import data_lib as dl

from collections import defaultdict
from math import ceil

def calculate_pad_length(window_size):
  assert window_size % 2 ==1
  return (window_size - 1)/2

def read_dataset_sentences(input_file, window_size):
  pad_length = calculate_pad_length(window_size)
  padding =  [(None, None)] * pad_length
  dataset = list(padding)
  with open(input_file, 'r') as f:
    for line in f:
      if line.strip():
        word, _, _, label = line.split()
        dataset.append((word, label))
      else:
        dataset.extend(list(padding))
  dataset+=list(padding)
  return dataset

def make_windows(dataset, window_size):
  windows = []
  pad_length = calculate_pad_length(window_size)
  for i, (word, label) in enumerate(dataset):
    if word is not None:
      window = [dataset[j][0] or "PAD" for j in range(i-pad_length,
                                             i+pad_length+1)]
      windows.append((window, label))
  return windows

def main():

  input_prefix, output_prefix, window_size = sys.argv[1:]
  window_size = int(window_size)

  train = read_dataset_sentences(input_prefix+'/train.txt', window_size)
  valid = read_dataset_sentences(input_prefix+'/dev.txt', window_size)
  train_windows = make_windows(train, window_size)
  valid_windows = make_windows(valid, window_size)
  
  labels = set()
  for dataset in [train_windows, valid_windows]:
    for _, label in dataset:
      labels.add(label)

  labels = sorted(list(labels))  
  label_map = dl.make_label_map(labels)

  for dataset, filename in zip([train_windows, valid_windows],
                                    ["train", "dev"]):
    out_file = output_prefix + filename + ".pickle"
    new_dataset = [(window, label_map[label]) for window, label in dataset]
    dl.make_pickle(new_dataset, label_map, out_file)
     
if __name__ == "__main__":
    main()
