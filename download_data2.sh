#!/bin/bash

mkdir data
cd data

7z e fma_metadata.7z

mkdir audio
cd audio

7z e fma_small.zip
