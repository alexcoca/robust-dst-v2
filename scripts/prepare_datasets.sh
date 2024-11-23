#!/bin/bash

# the project root
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
echo "Raw datasets will be put in $DIR/data/raw/"

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
cd $WORK_DIR/dstc8-schema-guided-dialogue-master
python3 -m venv venv/
source venv/bin/activate
which python3
pip3 install -r sgd_x/requirements.txt
python3 -m sgd_x.generate_sgdx_dialogues
deactivate
cd $DIR
mkdir -p $DIR/data/raw/original/dev
mkdir -p $DIR/data/raw/original/test
mkdir -p $DIR/data/raw/original/train
mv $WORK_DIR/dstc8-schema-guided-dialogue-master/sgd_x/data/* $DIR/data/raw/
mv $WORK_DIR/dstc8-schema-guided-dialogue-master/dev/* $DIR/data/raw/original/dev/
mv $WORK_DIR/dstc8-schema-guided-dialogue-master/test/* $DIR/data/raw/original/test/
mv $WORK_DIR/dstc8-schema-guided-dialogue-master/train/* $DIR/data/raw/original/train/

echo "Generating blank SGD dialogue templates"
declare -a versions=("original" "v1" "v2" "v3" "v4" "v5")
cd $DIR
for i in "${versions[@]}"
do
  mkdir -p data/interim/blank_dialogue_templates/"$i"/train
  python -m scripts.blank -i data/raw/"$i"/train -o data/interim/blank_dialogue_templates/"$i"/train
  mkdir -p data/interim/blank_dialogue_templates/"$i"/dev
  python -m scripts.blank -i data/raw/"$i"/dev -o data/interim/blank_dialogue_templates/"$i"/dev
  mkdir -p data/interim/blank_dialogue_templates/"$i"/test
  python -m scripts.blank -i data/raw/"$i"/test -o data/interim/blank_dialogue_templates/"$i"/test
done
