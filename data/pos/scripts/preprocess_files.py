import sys

def make_pos_map(path):
    pos_map = {}
    for line in open(path,'r'):
        (tag, u_tag) = line.split()
        if '|' not in tag:
            pos_map[tag] = u_tag
    return pos_map

def main():
  POS_map = make_pos_map(sys.argv[1])
  
  for out_name, in_name in zip(["train.txt", "dev.txt"],
                               ["train-wsj-0-18", "test-wsj-19-21"]):
    with open(in_name, 'r') as in_file:
      with open(out_name, 'w') as out_file:
        for line in in_file:
          for token in line.split():
            (word, tag) = token.rsplit("_",1)
            out_file.write(word.lower()+"\t"+POS_map[tag]+"\n")
        out_file.write("\n")

if __name__=="__main__":
    main()
