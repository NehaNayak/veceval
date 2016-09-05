import numpy as np
import pickle

def make_label_map(labels):
  label_map = {label:array 
               for label, array in zip(labels, np.eye(len(labels)))}
  return label_map

def make_pickle(dataset, label_map, out_file):
  pickle.dump((dataset, label_map), open(out_file,'w'))


