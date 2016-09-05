import sys
import data_lib as dl
import numpy as np
from math import ceil
from collections import Counter
import pickle
import random

def main():

  input_file, output_prefix = sys.argv[1:]
  
  dataset = []

  labels = ["1.0", "2.0", "3.0", "4.0", "5.0"]
  label_map = dl.make_label_map(labels)
  
  with open(input_file, 'r') as in_file:
    for line in in_file:
      (avg_score, _, p1, p2, _) = line.split("\t")
      p1_words = [word.lower() for word in p1.split()]
      p2_words = [word.lower() for word in p2.split()]
      label = str(ceil(float(avg_score)))
      if label == '0.0':
        label = '1.0'
      dataset.append((p1_words, p2_words, label_map[label]))

  random.Random(137).shuffle(dataset)

  train_len = len(dataset) - 4000
  dev_len = int(0.1 * float(train_len))

  dev = dataset[4000:(4000 + dev_len)]
  train = dataset[(4000 + dev_len):]
  
  for dataset, filename in zip([train, dev], ["train", "dev"]):
    out_file = output_prefix + filename + ".pickle"
    dl.make_pickle(dataset, label_map, out_file)

if __name__=="__main__":
  main()
