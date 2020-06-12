#!/bin/sh
# Parameters.

inputFolder=${1}
textFolder=${2}
outputFolder=${3}

python3 ./crop/croppedDemo.py ${inputFolder} ${textFolder} ${outputFolder}
