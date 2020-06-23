#!/bin/sh
# Parameters.

inputFolder=${1}
textFolder=${2}
outputFolder=${3}

python3 ./crop/croppedDemo.py ${inputFolder} ${textFolder} ${outputFolder}


python3 ./ocra/names.py
sort ./ocra/input_file.txt -o ./ocra/input_file.txt
python3 ./ocra/demo.py ./ocra/input_file.txt ./ocra/output_htk_format.txt