import importlib
import os

#os.system('python ./artextDetection/main/demo.py')
#print ('\n \n done detecting')
#os.system('python ./crop/cropped.py')
#print ('\n \n Done cropping')
os.system('python ./ocra/names.py')
os.system('sort ./ocra/input_file.txt -o ./ocra/input_file.txt')
os.system('python ./ocra/demo.py ./ocra/input_file.txt ./ocra/output_htk_format.txt')
print ('\n \n Done recognition')