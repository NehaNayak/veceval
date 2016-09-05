wget http://cogcomp.cs.illinois.edu/Data/QA/QC/train_5500.label
mv ./train_5500.label train.label
python make_datasets.py . ../
