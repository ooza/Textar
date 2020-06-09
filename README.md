# Arabic text detection and recognition in scene and video images
This work is dedicated to Arabic text recongition in multimedia documents, implemented using tensorflow and includes two parts: text regions localisation and textline recognition.

# Text detection based on CTPN
Arabic text detection in scene/video images based on ctpn (connectionist text proposal network). It is implemented in tensorflow. The origin paper can be found [here](https://arxiv.org/abs/1609.03605). Also, the origin repo in caffe can be found in [here](https://github.com/tianzhi0549/CTPN). For more detail about the paper and code, see this [blog](http://slade-ruan.me/2017/10/22/text-detection-ctpn/). If you got any questions, check the issue first, if the problem persists, open a new issue.
***
**NOTICE: Thanks to [banjin-xjy](https://github.com/banjin-xjy), which reimplemented the original code using Tensorflow.
***
## roadmap
- [x] reconstruct the repo
- [x] cython nms and bbox utils
- [x] loss function as referred in paper
- [x] oriented text connector
- [x] BLSTM
***
## setup
nms and bbox utils are written in cython, hence you have to build the library first.
```shell
cd utils/bbox
chmod +x make.sh
./make.sh
```
It will generate a nms.so and a bbox.so in current folder.
***
## demo
- follow setup to build the library 
- download the ckpt file from [googl drive](https://drive.google.com/file/d/1mky52CCr7g_fkjI9QHw6ZUg7clzsiVLT/view?usp=sharing)
- put checkpoints_mlt/ in Textar/
- put your images in data/demo, the results will be saved in data/res, and run demo in the root 
```shell
python3 ./artext_detection/main/demo.py
```
***
# OCR
## demo
- crop the input images based on the output detection coordinates
- save the cropped images' name in input_file.txt
- run demo in the root
```shell
python3 ./ocra/demo.py
```
***
**NOTICE: the training of this part is work in progress.
***
# End-to-end fashion
to run the code in an end-to-end fashion:
```shell
python3 ./run.py
```