# -*- coding: utf-8 -*-
import sys
import codecs
import pickle
import data_lib as dl

from collections import defaultdict
from math import ceil


def read_dataset_sentences(input_file):
  dataset = []
  with open(input_file, 'r') as f:
    for line in f:
      fields = line.split()
      label = fields[0].split(':')[0]
      question = [word.lower() for word in fields[1:]]
      dataset.append((question, label)) 
  return dataset
 

def main():

  input_prefix, output_prefix = sys.argv[1:]

  train = read_dataset_sentences(input_prefix+'/train.label')
  for dataset, filename in zip([train], ["train"]):
    out_file = output_prefix + filename + ".pickle"
    labels = sorted(list(set([label for question, label in dataset])))
    label_map = dl.make_label_map(labels)
    new_dataset = [(question, label_map[label]) for question, label in dataset]
    dl.make_pickle(new_dataset, label_map, out_file)
     
if __name__ == "__main__":
    main()
