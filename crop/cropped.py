from PIL import Image
import cv2
import numpy as np
import os

def crop(image_path, coords, saved_location):

    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """

    image_obj = Image.open(image_path)

    cropped_image = image_obj.crop(coords)

    cropped_image.save(saved_location)
    #cropped_image.show()
    
if __name__ == '__main__':
    exts = ("jpg", "png", "JPG")
    for img_file in os.listdir(os.path.abspath('data/demo/')):
        if img_file.endswith(exts):
            name=img_file[:-4]
            print(name)
            image_path= os.path.abspath('data/demo/'+name+'.png')
            txt_file_name=os.path.abspath('data/res/'+name+'.txt')
            #print ("filename",txt_file_name)
            file = open (txt_file_name, 'r')
            i=0
            for l in file:
                i=i+1
                m=l.split(',')
                m[8]=0
                m = [ int(x) for x in m ]
                #print(m[0],',',m[1],',',m[4],',',m[5] )
                name_output=name+'__'+str(i)+'.png'
                crop(image_path, (m[0],m[1]-8,m[2],m[5]-3 ),os.path.abspath('ocr/cropped_image/'+name_output) )
                #crop(image, (877,199,1130,1559),os.path.abspath('ocr/cropped_image/'+name_output) )
            file.close()
    