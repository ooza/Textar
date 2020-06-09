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
def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    print ('scale:' , im_scale)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    re_im.show()
    return re_im, (new_h / img_size[0], new_w / img_size[1])
if __name__ == '__main__':
    output_path='./Cropped_image'
    for i in os.listdir(os.path.abspath('data/demo/')):
        if i.endswith ('.jpg') == True:
            name=i[:-4]
            print(name)
            image = os.path.abspath('data/demo/'+name+'.jpg')
            print ("filename",image)
            txt_file_name=os.path.abspath('data/res/'+name+'.txt')
            print ("filename",txt_file_name)
            file = open (txt_file_name, 'r')
            i=0
            for l in file:
                i=i+1
                m=l.split(',')
                m[8]=0
                m = [ int(x) for x in m ]
                print(m[0],',',m[1],',',m[4],',',m[5] )
                name_output=name+'__'+str(i)+'.jpg'
                crop(image, (m[0],m[1]-8,m[4],m[5]-3),os.path.abspath('ocr/cropped_image/'+name_output) )
                #image.crop((m[0],m[1]-8,m[4],m[5]-3)).save(os.path.abspath('ocr/cropped_image/'+name_output), quality=95)
            file.close()
