curl -O http://cogcomp.org/Data/QA/QC/train_5500.label
mv ./train_5500.label train.label
python make_datasets.py . ../
