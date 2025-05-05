#!/bin/bash

# Check if DATASET_DIRECTORY is already set, if not set it to the default value
if [ -z "$DATASET_DIRECTORY" ]; then
  export DATASET_DIRECTORY="/scratch/mohanty/mapping-challenge-data/raw"
fi

mkdir -p $DATASET_DIRECTORY

# Download files sequentially
echo "Downloading Vegas dataset..."
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/train/tarballs/SN2_buildings_train_AOI_2_Vegas.tar.gz $DATASET_DIRECTORY/

echo "Downloading Paris dataset..."
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/train/tarballs/SN2_buildings_train_AOI_3_Paris.tar.gz $DATASET_DIRECTORY/

echo "Downloading Shanghai dataset..."
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/train/tarballs/SN2_buildings_train_AOI_4_Shanghai.tar.gz $DATASET_DIRECTORY/

echo "Downloading Khartoum dataset..."
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/train/tarballs/SN2_buildings_train_AOI_5_Khartoum.tar.gz $DATASET_DIRECTORY/

cd $DATASET_DIRECTORY

# Extract files sequentially
echo "Extracting Vegas dataset..."
tar -xvzf SN2_buildings_train_AOI_2_Vegas.tar.gz

echo "Extracting Paris dataset..."
tar -xvzf SN2_buildings_train_AOI_3_Paris.tar.gz

echo "Extracting Shanghai dataset..."
tar -xvzf SN2_buildings_train_AOI_4_Shanghai.tar.gz

echo "Extracting Khartoum dataset..."
tar -xvzf SN2_buildings_train_AOI_5_Khartoum.tar.gz
