#!/bin/bash

# Download rio dataset from s3://spacenet-dataset/competition1
# mkdir -p competition1/spacenet_TrainData
# mkdir -p competition1/spacenet_TrainData/vectordata
# aws s3api get-object --bucket spacenet-dataset --key competition1/spacenet_TrainData/3band.tar.gz --request-payer requester competition1/spacenet_TrainData/3band.tar.gz &
# aws s3api get-object --bucket spacenet-dataset --key competition1/spacenet_TrainData/vectordata/geojson.tar.gz --request-payer requester competition1/spacenet_TrainData/vectordata/geojson.tar.gz & 
# aws s3api get-object --bucket spacenet-dataset --key competition1/spacenet_TrainData/vectordata/summarydata.tar.gz --request-payer requester competition1/spacenet_TrainData/vectordata/summarydata.tar.gz &


# Spacenet V2 dataset
# https://spacenet.ai/spacenet-buildings-dataset-v2/

# Shanghai
mkdir -p data/AOI_4_Shanghai/
aws s3api get-object --bucket spacenet-dataset --key AOI_4_Shanghai/AOI_4_Shanghai_Train.tar.gz --request-payer requester data/AOI_4_Shanghai/AOI_4_Shanghai_Train.tar.gz &
aws s3api get-object --bucket spacenet-dataset --key AOI_4_Shanghai/AOI_4_Shanghai_Test_public.tar.gz --request-payer requester data/AOI_4_Shanghai/AOI_4_Shanghai_Test_public.tar.gz &

# Vegas
mkdir -p data/AOI_2_Vegas/
aws s3api get-object --bucket spacenet-dataset --key AOI_2_Vegas/AOI_2_Vegas_Train.tar.gz --request-payer requester data/AOI_2_Vegas/AOI_2_Vegas_Train.tar.gz &
aws s3api get-object --bucket spacenet-dataset --key AOI_2_Vegas/AOI_2_Vegas_Test_public.tar.gz --request-payer requester data/AOI_2_Vegas/AOI_2_Vegas_Test_public.tar.gz &

#Paris
mkdir -p data/AOI_3_Paris/
aws s3api get-object --bucket spacenet-dataset --key AOI_3_Paris/AOI_3_Paris_Train.tar.gz --request-payer requester data/AOI_3_Paris/AOI_3_Paris_Train.tar.gz & 
aws s3api get-object --bucket spacenet-dataset --key AOI_3_Paris/AOI_3_Paris_Test_public.tar.gz --request-payer requester data/AOI_3_Paris/AOI_3_Paris_Test_public.tar.gz &

#Khartoum
mkdir -p data/AOI_5_Khartoum/
aws s3api get-object --bucket spacenet-dataset --key AOI_5_Khartoum/AOI_5_Khartoum_Train.tar.gz --request-payer requester data/AOI_5_Khartoum/AOI_5_Khartoum_Train.tar.gz &
aws s3api get-object --bucket spacenet-dataset --key AOI_5_Khartoum/AOI_5_Khartoum_Test_public.tar.gz --request-payer requester data/AOI_5_Khartoum/AOI_5_Khartoum_Test_public.tar.gz &



#SpacenetRoads
# aws s3api get-object --bucket spacenet-dataset --key SpaceNet_Roads_Competition/AOI_2_Vegas_Roads_Test_Public.tar.gz --request-payer requester AOI_2_Vegas_Roads_Test_Public.tar.gz &
# aws s3api get-object --bucket spacenet-dataset --key SpaceNet_Roads_Competition/AOI_2_Vegas_Roads_Train.tar.gz --request-payer requester AOI_2_Vegas_Roads_Train.tar.gz &
# aws s3api get-object --bucket spacenet-dataset --key SpaceNet_Roads_Competition/AOI_3_Paris_Roads_Test_Public.tar.gz --request-payer requester AOI_3_Paris_Roads_Test_Public.tar.gz &
# aws s3api get-object --bucket spacenet-dataset --key SpaceNet_Roads_Competition/AOI_3_Paris_Roads_Train.tar.gz --request-payer requester AOI_3_Paris_Roads_Train.tar.gz & 
# aws s3api get-object --bucket spacenet-dataset --key SpaceNet_Roads_Competition/AOI_4_Shanghai_Roads_Test_Public.tar.gz --request-payer requester AOI_4_Shanghai_Roads_Test_Public.tar.gz &
# aws s3api get-object --bucket spacenet-dataset --key SpaceNet_Roads_Competition/AOI_4_Shanghai_Roads_Train.tar.gz --request-payer requester AOI_4_Shanghai_Roads_Train.tar.gz &
# aws s3api get-object --bucket spacenet-dataset --key SpaceNet_Roads_Competition/AOI_5_Khartoum_Roads_Test_Public.tar.gz --request-payer requester AOI_5_Khartoum_Roads_Test_Public.tar.gz &
# aws s3api get-object --bucket spacenet-dataset --key SpaceNet_Roads_Competition/AOI_5_Khartoum_Roads_Train.tar.gz --request-payer requester AOI_5_Khartoum_Roads_Train.tar.gz &








