# crowdAI-mapping-challenge-preprocessing
![CrowdAI-Logo](https://github.com/crowdAI/crowdai/raw/master/app/assets/images/misc/crowdai-logo-smile.svg?sanitize=true)

**TODO**: Add introduction, installation, usage

# Install 
```
# This is a fork of the spacenetv2 utilities repo, which adds the py3 support + parallelization using ray
git clone git@github.com:spMohanty/spacenetv2-utilities-py3.git utilities
cd utilities 
pip install -e .

cd ../
git clone git@github.com:spMohanty/mapping-challenge-preprocessing.git
cd mapping-challenge-preprocessing
pip install -r requirements.txt
```

# Usage
```
./download_data.sh # please refer to the inline comments to untar the files

# generte the RGB data and the PASCALVOC annotations
export DATASET_DIRECTORY="/scratch/mohanty/mapping-challenge-data" # update this path to your local path

DATASET_NAME="AOI_2_Vegas_Train" ./prepare-data.sh
DATASET_NAME="AOI_3_Paris_Train" ./prepare-data.sh
DATASET_NAME="AOI_4_Shanghai_Train" ./prepare-data.sh
DATASET_NAME="AOI_5_Khartoum_Train" ./prepare-data.sh


# python main.py
```

# Authors
* Iuliana Voinea <iulianavoinea96@gmail.com>
* Snigdha Dagar <snigdha.dagar@gmail.com>
* Florian Laurent <florian.laurent@gmail.com>
* Sharada Mohanty <sharada.mohanty@epfl.ch>
