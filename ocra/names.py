import os

trt=os.listdir("./ocra/cropped_image")
file=open('./ocra/input_file_'+trt[0]+'.txt', 'w')
for i in trt:
    name=i
    file.write('./ocra/cropped_image/'+name+'\n')
file.close()
