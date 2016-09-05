#!/bin/bash

# Error on a failed command
set -o errexit

# Get the root of the repository
ME="${BASH_SOURCE[0]}"
MYDIR=$( cd "$( dirname "$ME" )" && pwd )
export ROOTDIR="$MYDIR"

# Setup

export AFFILIATION="VecEval" # Ensure affiliation uses only a-z, A-Z, 0-9
if ! [[ $AFFILIATION =~ ^[A-Za-z]+$ ]]; then
  echo "Error: Please choose an affiliation using only letters."
  exit
fi

# Uncomment the line below and replace the embedding home path
export EMBEDDINGS_HOME="/media/embeddings/" # directory containing gzipped embedding files
export LOG_FILE=$ROOTDIR"/LOG"
export CHECKPOINT_HOME="/tmp/veceval/checkpoints/"
export PICKLES_HOME="/tmp/veceval/pickles/"

mkdir -p $CHECKPOINT_HOME
mkdir -p $PICKLES_HOME

# Create a virtualenv if it doesn't exist,
# and if a custom $PYTHON hasn't been set
if [ -z "$PYTHON" ]; then
  VIRTUALENV="$ROOTDIR/veceval_env"
  if [ ! -d "$VIRTUALENV" ]; then
    virtualenv -p "`which python2.7`" "$VIRTUALENV"
    source "$VIRTUALENV/bin/activate"
  else
    source "$VIRTUALENV/bin/activate"
  fi
  
  # Install python dependencies
  PYTHON="$VIRTUALENV/bin/python2.7"
  PIP="$VIRTUALENV/bin/pip"
  export LC_ALL=C
  pip install -q -r "$ROOTDIR/requirements.txt"
  echo "USING PYTHON: $PYTHON"
else
  echo "USING PYTHON: $PYTHON"
fi

echo $ROOTDIR

# From here on, force all variables to be defined
set -o nounset

# Prepare embeddings
export EMBEDDING_NAME="$1"
if ! [[ $1 =~ ^[A-Za-z0-9]+$ ]]; then
  echo "Error: Please choose an embedding name using only letters and digits."
  exit
fi
export EMBEDDING_NAME=$AFFILIATION"_""$1"

python \
  $ROOTDIR"/embeddings/prepare_embeddings.py" \
  $EMBEDDINGS_HOME$EMBEDDING_NAME".txt.gz" \
  $ROOTDIR"/embeddings/common_vocabulary.txt" \
  $PICKLES_HOME$EMBEDDING_NAME".pickle"

for task in sentiment questions ner chunk #nli pos
do
  for with_backprop in finetuned fixed
  do
    TRAIN_SCRIPT=$ROOTDIR"/training/"$task"_"$with_backprop".py"
    CONFIG_FILE=$ROOTDIR"/training/configs/config_"$task"_"$with_backprop".txt"
    python $TRAIN_SCRIPT $CONFIG_FILE $EMBEDDING_NAME
  done
done

deactivate
