import os
file=open('./ocr/input_file.txt', 'w')
for i in os.listdir("./ocr/cropped_image"):
    name=i
    file.write('./ocr/cropped_image/'+name+'\n')
file.close()
