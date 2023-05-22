#!/bin/bash

FILE=./data

if test -d "$FILE"; then
    echo "Data file already exists"
else
    mkdir data
    cd data

    curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
    #echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -

    7z e fma_metadata.zip

    mkdir audio
    cd audio

    curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
    #echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | sha1sum -c -

    7z e fma_small.zip
fi