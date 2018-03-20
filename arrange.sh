rm -rf /mount/SDG/mapping-challenge/AOI_2_Vegas_Train/PASCALVOC_annotations; \
/usr/bin/python ../utilities/python/createDataSpaceNet.py /mount/SDG/mapping-challenge/AOI_2_Vegas_Train/ \
           --srcImageryDirectory RGB-PanSharpen \
           --outputDirectory /mount/SDG/mapping-challenge/AOI_2_Vegas_Train/PASCALVOC_annotations \
           --annotationType PASCALVOC2012 \
           --outputFileType JPEG \
           --convertTo8Bit \
           --imgSizePix 300 &

rm -rf /mount/SDG/mapping-challenge/AOI_3_Paris_Train/PASCALVOC_annotations; \
/usr/bin/python ../utilities/python/createDataSpaceNet.py /mount/SDG/mapping-challenge/AOI_3_Paris_Train/ \
          --srcImageryDirectory RGB-PanSharpen \
          --outputDirectory /mount/SDG/mapping-challenge/AOI_3_Paris_Train/PASCALVOC_annotations \
          --annotationType PASCALVOC2012 \
          --outputFileType JPEG \
          --convertTo8Bit \
          --imgSizePix 300 &

rm -rf /mount/SDG/mapping-challenge/AOI_4_Shanghai_Train/PASCALVOC_annotations; \
/usr/bin/python ../utilities/python/createDataSpaceNet.py /mount/SDG/mapping-challenge/AOI_4_Shanghai_Train/ \
          --srcImageryDirectory RGB-PanSharpen \
          --outputDirectory /mount/SDG/mapping-challenge/AOI_4_Shanghai_Train/PASCALVOC_annotations \
          --annotationType PASCALVOC2012 \
          --outputFileType JPEG \
          --convertTo8Bit \
          --imgSizePix 300 &

rm -rf /mount/SDG/mapping-challenge/AOI_5_Khartoum_Train/PASCALVOC_annotations; \
/usr/bin/python ../utilities/python/createDataSpaceNet.py /mount/SDG/mapping-challenge/AOI_5_Khartoum_Train/ \
          --srcImageryDirectory RGB-PanSharpen \
          --outputDirectory /mount/SDG/mapping-challenge/AOI_5_Khartoum_Train/PASCALVOC_annotations \
          --annotationType PASCALVOC2012 \
          --outputFileType JPEG \
          --convertTo8Bit \
          --imgSizePix 300 &
