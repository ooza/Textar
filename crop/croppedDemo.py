from PIL import Image
import cv2
import numpy as np
import os
import argparse



parser = argparse.ArgumentParser(description="Demo")
parser.add_argument('image', type=str, help='Input image path')
parser.add_argument('bbox_coord', type=str, help='textfile coordinates path')

parser.add_argument('cropped_img', type=str, help='Output cropped images path')

args = parser.parse_args()


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
    for img_file in os.listdir(os.path.abspath(args.image)):
        if img_file.endswith(exts):
            name=img_file[:-4]
            print(name)
            image_path= os.path.abspath(args.image+name+'.png')
            txt_file_name=os.path.abspath(args.bbox_coord+name+'.txt')

            file = open (txt_file_name, 'r')
            i=0
            for l in file:
                i=i+1
                m=l.split(',')
                m[8]=0
                m = [ int(x) for x in m ]
                #print(m[0],',',m[1],',',m[4],',',m[5] )

                name_output=name+'__'+str(i)+'.png'

                crop(image_path, (m[0],m[1]-8,m[2],m[5]-3 ),os.path.abspath(args.cropped_img+name_output) )
                #crop(image, (877,199,1130,1559),os.path.abspath('ocra/cropped_image/'+name_output) )
            file.close()
    