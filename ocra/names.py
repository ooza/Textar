import os
file=open('./ocra/input_file.txt', 'w')
for i in os.listdir("./ocra/cropped_image"):
    name=i
    file.write('./ocra/cropped_image/'+name+'\n')
file.close()
