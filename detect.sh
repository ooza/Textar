#!/bin/sh
# Parameters.

inputFolder=${1}
outputFolder=${2}

python3 ./artext_detection/main/demoDetection.py ${inputFolder} ${outputFolder}