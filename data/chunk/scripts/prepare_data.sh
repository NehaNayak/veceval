wget http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz
gunzip train.txt.gz

python make_datasets.py ./ ../ 5
