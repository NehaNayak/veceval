scp nayakne@jacob.stanford.edu:/u/nlp/data/ner/conll/eng.train ./train.txt
scp nayakne@jacob.stanford.edu:/u/nlp/data/ner/conll/eng.testa ./dev.txt

python make_datasets.py ./ ../ 5
