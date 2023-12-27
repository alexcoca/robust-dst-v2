#!/bin/bash

# the project root
FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../.. && pwd )"
echo "Raw datasets will be put in $DIR/data/raw/"

# the temp directory used, within $DIR
# omit the -p parameter to create a temporal directory in the default location
WORK_DIR=`mktemp -d -p "$DIR"`

# The directory where the raw SGD data will be placed
raw_dir=$DIR/data/raw/SGD
mkdir -p $raw_dir

# check if tmp dir was created
if [[ ! "$WORK_DIR" || ! -d "$WORK_DIR" ]]; then
  echo "Could not create temp dir"
  exit 1
fi

# deletes the temp directory
function cleanup {
  rm -rf "$WORK_DIR"
  echo "Deleted temp working directory $WORK_DIR"
}

# register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

# SGD-X
wget --directory-prefix=$WORK_DIR https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/archive/refs/heads/master.zip
unzip $WORK_DIR/master.zip -d $WORK_DIR
mv $WORK_DIR/dstc8-schema-guided-dialogue-master/* $raw_dir

python_script=$FILE_DIR/create_sgd_sdt.py
interim_dir=$DIR/data/interim/SGD_SDT
mkdir -p $interim_dir

for subdir in "train" "test" "dev"
do
  for indices in "0" "1" "2" "3" "4"
  do
    echo $subdir-$indices
    output_path=$interim_dir/v"$indices"_"$subdir".tsv
    echo $output_path
    python $python_script --input_dir=$raw_dir --output_path=$output_path --prompt_indices=$indices \
    --mcq_cat_vals --subdirs=$subdir
  done
done
