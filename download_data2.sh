#!/bin/bash

mkdir data
cd data

7za e fma_metadata.7z

mkdir audio
cd audio

7za e fma_small.zip
