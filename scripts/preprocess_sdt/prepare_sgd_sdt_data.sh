#!/bin/bash

# the project root
FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../.. && pwd )"

# The directory where the raw SGD data will be placed
raw_dir=$DIR/data/raw/SGD
mkdir -p $raw_dir

echo "Raw datasets will be put in $raw_dir"

# the temp directory used, within $DIR
# omit the -p parameter to create a temporal directory in the default location
WORK_DIR=`mktemp -d -p "$DIR"`

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

create_script=$FILE_DIR/create_sgd_sdt.py
preprocess_script=$FILE_DIR/preprocess_sdt_sgd.py

interim_dir=$DIR/data/interim/SGD_SDT
processed_dir=$DIR/data/processed/SGD_SDT

preprocessing_config=$DIR/configs/data_processing_sdt_sgd.yaml

mkdir -p $interim_dir
mkdir -p $processed_dir

for subdir in "train" "test" "dev"
do
  for indices in "0" "1" "2" "3" "4"
  do
    echo $subdir-$indices
    data_create_path=$interim_dir/v"$indices"_"$subdir".tsv
    echo $data_create_path
    python $create_script --input_dir=$raw_dir --output_path=$data_create_path --prompt_indices=$indices \
    --mcq_cat_vals --subdirs=$subdir
    output_path=$processed_dir/v"$indices/$subdir"
    python $preprocess_script --data_path=$data_create_path --raw_data_path=$raw_dir \
    --prompt_indices=$indices --output_path=$output_path --config=$preprocessing_config
  done
done

# Delete Interim Folder after processing is done
rm -rf $interim_dir
