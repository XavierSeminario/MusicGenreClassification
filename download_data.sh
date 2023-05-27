#!/bin/bash

FILE=./data
# download and extract two zip files ("fma_metadata.zip" and "fma_small.zip") from the provided URLs 
#It checks if a directory named "data" already exists and if not, creates it and performs the download and extraction operations within the "data" directory.
if test -d "$FILE"; then
    echo "Data file already exists"
else
    mkdir data
    cd data

    curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
    #echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -

    7z e fma_metadata.zip

    mkdir audio
    mkdir Spectrograms
    cd audio

    curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
    #echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | sha1sum -c -

    7z e fma_small.zip
    rm -rf README.txt
    rm -rf checksums
    rm -rf fma_small.zip
fi
