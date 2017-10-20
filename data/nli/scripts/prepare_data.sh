curl -O http://www.seas.upenn.edu/~nlp/resources/ppdb-2.0-human-labels.tgz
tar -zxf ppdb-2.0-human-labels.tgz
mv human-labeled-data/ppdb-sample.tsv ./
rm -r human-labeled-data/
rm ppdb-2.0-human-labels.tgz

python make_datasets.py ./ppdb-sample.tsv ../
