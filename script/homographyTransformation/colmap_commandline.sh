#!/bin/bash
# This is a script for COLMAP georegistration and then run dense reconstruction
####

CURRENT=$(cd $(dirname $0);pwd)
# source /home/yunakato/SingleViewSurgicalVideo/ini.txt
source $CURRENT/ini.txt

mkdir $DATASET_PATH/000/images
cp $DATASET_PATH/000/*f.jpg $DATASET_PATH/000/images

colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

# mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/000/images \
    --output_path $DATASET_PATH/000

colmap model_converter \
    --input_path $DATASET_PATH/000/0 \
    --output_path $DATASET_PATH/000 \
    --output_type TXT


# getHomography2.py
# python /home/yunakato/STEALTHs-3d-video/getHomography2.py $DATASET_PATH
