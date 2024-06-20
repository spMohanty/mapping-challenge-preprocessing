#!/bin/bash



# Check if DATASET_DIRECTORY is already set, if not set it to the default value
if [ -z "$DATASET_DIRECTORY" ]; then
  export DATASET_DIRECTORY="/scratch/mohanty/mapping-challenge-data/raw"
fi

# Check if DATASET_NAME is already set, if not set it to the default value
if [ -z "$DATASET_NAME" ]; then
  export DATASET_NAME="AOI_2_Vegas_Train"
fi

export ANNOTATIONS_DIRECTORY="$DATASET_DIRECTORY/$DATASET_NAME/PASCALVOC_annotations"

if [ -d $ANNOTATIONS_DIRECTORY ]; then
  echo "Annotations Directory exists already. Deleting it before proceeding: $ANNOTATIONS_DIRECTORY."
  rm -rf $ANNOTATIONS_DIRECTORY
fi

export PYTHONPATH="$PYTHONPATH:../utilities/python"

python ../utilities/python/createDataSpaceNet.py $DATASET_DIRECTORY/$DATASET_NAME/ \
           --srcImageryDirectory RGB-PanSharpen \
           --outputDirectory $ANNOTATIONS_DIRECTORY \
           --annotationType PASCALVOC2012 \
           --outputFileType JPEG \
           --convertTo8Bit \
           --imgSizePix 300 \
           --trainTestSplit 0.8
