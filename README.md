# VecEval

This is the code for the evaluation at [veceval.com](http://www.veceval.com).

To run the evaluation, first run the preparation script:

```
bash veceval_prepare.sh
```

You will have to modify some lines in `data/ner/scripts/prepare_data.sh` and `/data/pos/scripts/prepare_data.sh` to access your version of NER and POS training data.

Then, run the evaluation script:

```
bash veceval_evaluate.sh embedding_name /path/to/embeddings.txt.gz
```

Please contact nayakne@cs.stanford.edu with any questions!
