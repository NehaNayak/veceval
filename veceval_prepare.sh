#!/bin/bash

# Error on a failed command
set -o errexit

# Get the root of the repository
ME="${BASH_SOURCE[0]}"
MYDIR=$( cd "$( dirname "$ME" )" && pwd )
export ROOTDIR="$MYDIR"

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

for task in sentiment questions ner chunk nli pos
do
  echo "Preparing data for $task"
  TRAIN_LOCATION=$ROOTDIR"/data/"$task"/scripts/"
  cd $TRAIN_LOCATION
  bash prepare_data.sh
done

deactivate
