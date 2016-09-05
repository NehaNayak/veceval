# -*- coding: utf-8 -*-
import sys
import codecs
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
        word, _, label = line.split()
        dataset.append((word.lower(), label))
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
  train_windows = make_windows(train, window_size)
  
  train_labels = sorted(list(set([label for window, label in train_windows])))
  train_labels.append('I-LST')  # Missing from training set
  labels = sorted(set(train_labels))
  print labels
  label_map = dl.make_label_map(labels)

  for dataset, filename in zip([train_windows],
                                    ["train"]):
    out_file = output_prefix + filename + ".pickle"
    new_dataset = [(window, label_map[label]) for window, label in dataset]
    dl.make_pickle(new_dataset, label_map, out_file)
     
if __name__ == "__main__":
    main()
