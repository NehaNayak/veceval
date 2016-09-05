import pickle
import veceval as ve
import sys
import gzip
import numpy as np

def main():
  embeddings_file, common_vocab_file, output_file = sys.argv[1:]

  common_vocabulary = set()
  for line in open(common_vocab_file, 'r'):
    common_vocabulary.add(line.strip())
 
  embedding_dict = {}
  unk_vectors = []

  with gzip.open(embeddings_file, 'r') as embedding_file:
    for line in embedding_file:
      this_line = line.split()
      assert len(this_line) == 51
      word = this_line[0].lower()
      vector = np.array([float(x) for x in this_line[1:]])
      if word in common_vocabulary:
        embedding_dict[word] = vector
      else:
        unk_vectors.append(vector)

  embedding_dict[ve.PAD] = np.zeros(embedding_dict.values()[0].shape)
  if unk_vectors:
    embedding_dict[ve.UNK] = sum(unk_vectors)/len(unk_vectors)
  else:
    embedding_dict[ve.UNK] = np.zeros(embedding_dict.values()[0].shape)

  with open(output_file, 'w') as output_file:
    pickle.dump(embedding_dict, output_file)


if __name__ == "__main__":
  main()
