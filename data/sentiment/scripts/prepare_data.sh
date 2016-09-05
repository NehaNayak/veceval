wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
unzip stanfordSentimentTreebank.zip 
rm -r __MACOSX/
rm stanfordSentimentTreebank.zip 
python make_datasets.py stanfordSentimentTreebank ../
rm -r stanfordSentimentTreebank
