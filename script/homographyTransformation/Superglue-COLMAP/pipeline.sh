#!/bin/bash

echo "Running the entire pipeline"

CURRENT=$(cd $(dirname $0);pwd)
# source /home/yunakato/SingleViewSurgicalVideo/ini.txt
source $CURRENT/../ini.txt

FRAME_DIRS=$PROJ_DIR$DATA_DIR/"???"

for FRAME_DIR in $FRAME_DIRS
do
	python $PROJ_DIR/imgs2pairs.py $FRAME_DIR $FRAME_DIR/pairs.txt
	python $SUPERGLUE_DIR/match_pairs.py --input_pairs $FRAME_DIR/pairs.txt --input_dir $FRAME_DIR --output_dir $FRAME_DIR/dump_match_pairs --resize -1 -1 --viz
done

python $PROJ_DIR/example_frames.py $PROJ_DIR/$DATA_DIR --database_path $PROJ_DIR/$DATA_DIR/database.db
