#!/bin/sh
# Parameters.

inputFolder=${1}
outputFolder=${2}

python3 ./artextDetection/main/demoDetection.py ${inputFolder} ${outputFolder}
#python3 artextDetection/main/demoDetection.py ${inputFolder} ${outputFolder}