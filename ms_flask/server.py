from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import os
from uuid import uuid4
import base64
import importlib

 	
import shutil
from shutil import copyfile
# Initialize the Flask application
app = Flask(__name__)


@app.route('/')
def hello():
        return "Hello World!"


# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():

    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #cv2.imshow('image',img)
    cv2.imwrite('data/demo/hi.png',img)
    
    # do some fancy processing here....
    os.system('python ./run.py')
    '''
    file= open ('./ocr/resulttxt.txt', 'r')
    lines= file.readlines()
    t=''
    for line in lines:
        t=t+'/'+line
    file.close()
    '''
    shutil.copy('/media/khalyl/b19f6211-f6a7-443d-8a50-5c247986129e/khalyl/Desktop/image_recog/text-detection-ctpn-banjin-dev/ocr/resulttxt.txt', '/media/khalyl/b19f6211-f6a7-443d-8a50-5c247986129e/khalyl/Desktop/work/tacotron/arabic-tacotron-tts/Shakkala-master/text.txt')
    #Generating waveform
    os.system ('/media/khalyl/b19f6211-f6a7-443d-8a50-5c247986129e/khalyl/Desktop/work/tacotron/arabic-tacotron-tts/Shakkala-master/demo.py')
    os.system('python /media/khalyl/b19f6211-f6a7-443d-8a50-5c247986129e/khalyl/Desktop/work/tacotron/arabic-tacotron-tts/demo.py')
    # build a response dict to send back to client
    #response = 'message : image received. size={}x{}'.format(img.shape[1], img.shape[0])+'\n' + t
########################################################################    
    #response audio file:
    shutil.copy('/media/khalyl/b19f6211-f6a7-443d-8a50-5c247986129e/khalyl/Desktop/work/tacotron/arabic-tacotron-tts/output/test.wav', "/media/khalyl/b19f6211-f6a7-443d-8a50-5c247986129e/khalyl/Desktop/image_recog/text-detection-ctpn-banjin-dev/test.wav")
    
    data = open("test.wav", "rb")
    encoded_f1 = base64.b64encode(data.read())
    print (type(encoded_f1))
    #print (encoded_f1)
    b2str=encoded_f1.decode("utf-8")
    print (type(b2str))
    response = b2str           
    
#######################################################################    
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    print ('image recieved')
    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)