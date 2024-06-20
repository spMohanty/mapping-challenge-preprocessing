#!/bin/bash

# Check if DATASET_DIRECTORY is already set, if not set it to the default value
if [ -z "$DATASET_DIRECTORY" ]; then
  export DATASET_DIRECTORY="/scratch/mohanty/mapping-challenge-data/raw"
fi

mkdir -p $DATASET_DIRECTORY

# Create a list of download links and file paths
DOWNLOAD_LINKS=(
  "s3://spacenet-dataset/spacenet/SN2_buildings/train/tarballs/SN2_buildings_train_AOI_2_Vegas.tar.gz"
  "s3://spacenet-dataset/spacenet/SN2_buildings/train/tarballs/SN2_buildings_train_AOI_3_Paris.tar.gz"
  "s3://spacenet-dataset/spacenet/SN2_buildings/train/tarballs/SN2_buildings_train_AOI_4_Shanghai.tar.gz"
  "s3://spacenet-dataset/spacenet/SN2_buildings/train/tarballs/SN2_buildings_train_AOI_5_Khartoum.tar.gz"
)

FILE_PATHS=(
  "$DATASET_DIRECTORY/SN2_buildings_train_AOI_2_Vegas.tar.gz"
  "$DATASET_DIRECTORY/SN2_buildings_train_AOI_3_Paris.tar.gz"
  "$DATASET_DIRECTORY/SN2_buildings_train_AOI_4_Shanghai.tar.gz"
  "$DATASET_DIRECTORY/SN2_buildings_train_AOI_5_Khartoum.tar.gz"
)

# Download files in parallel
parallel -j 4 aws s3 cp ::: "${DOWNLOAD_LINKS[@]}" ::: "${FILE_PATHS[@]}"

cd $DATASET_DIRECTORY

# Create a list of tar files to extract
TAR_FILES=(
  "SN2_buildings_train_AOI_2_Vegas.tar.gz"
  "SN2_buildings_train_AOI_3_Paris.tar.gz"
  "SN2_buildings_train_AOI_4_Shanghai.tar.gz"
  "SN2_buildings_train_AOI_5_Khartoum.tar.gz"
)

# Extract tar files in parallel
parallel -j 4 tar -xvzf ::: "${TAR_FILES[@]}"
