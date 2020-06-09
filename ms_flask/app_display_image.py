import os
from uuid import uuid4

import importlib
 	
import shutil

from flask import Flask, request, render_template, send_from_directory

__author__ = 'ibininja'
PEOPLE_FOLDER = os.path.join('static','images')
app = Flask(__name__)
# app = Flask(__name__, static_folder="images")
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/images')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #print (full_filename)
    src='/media/khalyl/b19f6211-f6a7-443d-8a50-5c247986129e/khalyl/Desktop/image_recog/text-detection-ctpn-banjin-dev/static/images/'+filename
    dst='/media/khalyl/b19f6211-f6a7-443d-8a50-5c247986129e/khalyl/Desktop/image_recog/text-detection-ctpn-banjin-dev/data/demo/'+ filename
    dst1='/media/khalyl/b19f6211-f6a7-443d-8a50-5c247986129e/khalyl/Desktop/image_recog/text-detection-ctpn-banjin-dev/data/res/'+ filename
    shutil.copy(src, dst)
    os.system('python ./run.py')
    shutil.copy(dst1, src)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file= open ('./ocr/resulttxt.txt', 'r')
    lines= file.readlines()
    t=''
    for line in lines:
        t=t+'/'+line
    file.close()
    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete.html", user_image=full_filename , text=t) 
'''
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)
'''
if __name__ == "__main__":
    app.run(port=4555, debug=True)
