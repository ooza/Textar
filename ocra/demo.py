from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
import os, codecs, cv2, sys, numpy as np

def enhance(img, low, high, extreme=0):
    bright = np.amax(img)
    dark = np.amin(img)
    low_in = dark + (bright - dark) * low
    high_in = dark + (bright - dark) * high
    if extreme == 0:
        img[img < low_in] = dark
        img[img > high_in] = bright
    else:
        img[img < low_in] = 0
        img[img > high_in] = 255
    return img


def ifreverse1(img):
    height = img.shape[0]
    width = img.shape[1]
    top = img[0, :]
    avg_edge = np.mean(top)
    center_row = img[int(round(height / 2)), :]
    avg_center = np.mean(center_row)
    if avg_edge > avg_center:
        img = 255 - img
    return img


def ifreverse2(img):
    row = img[2, :]
    large = np.sum(img == 255)
    small = np.sum(img == 0)
    if large > small:
        img = 255 - img
    return img


def ifreverse3(img):
    height = img.shape[0]
    width = img.shape[1]
    if img[(0, 0)] == 255 and img[(0, width - 1)] == 255 and img[(height - 1, 0)] == 255 and img[(height - 1, width - 1)] == 255:
        img = 255 - img
    return img


def preprocess(img_path):
    img = cv2.imread(img_path)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayimg = enhance(grayimg, 0.4, 0.6, 0)
    grayimg = ifreverse1(grayimg)
    grayimg = enhance(grayimg, 0.3, 0.7, 1)
    grayimg = ifreverse2(grayimg)
    grayimg = ifreverse3(grayimg)
    dims = (int(round(48 * grayimg.shape[1] / grayimg.shape[0])), 48)
    grayimg = cv2.resize(grayimg, dims)
    grayimg = np.asarray(grayimg, dtype=np.uint32)
    return grayimg


def img2input(image):
    dims = np.shape(image)
    image = image[:, ::-1]
    input = np.zeros(shape=[1, dims[1], 48, 1])
    input[0, :dims[1], :, 0] = np.transpose(image[np.newaxis, :, :], (0, 2, 1))
    len = np.zeros(shape=[1], dtype='int64')
    len[0] = dims[1]
    return (
     input / 255.0, len)


def get_row(sparse_tuple, row, dtype=np.int32):
    optlist = []
    cnt = 0
    for pos in sparse_tuple[0]:
        if pos[0] == row:
            optlist.append(sparse_tuple[1][cnt])
        cnt += 1

    return optlist


class Model(object):

    def __init__(self):
        image_height = 48
        num_classes = 169
        num_hidden_1 = 50
        num_hidden_2 = 100
        num_hidden_3 = 200
        num_hidden_4 = 200
        self.inputs0 = tf.placeholder(tf.float32, [None, None, image_height, 1], name='inputs')
        inputs = tf.reshape(self.inputs0, [tf.shape(self.inputs0)[0], -1, image_height])
        inputs = (inputs - 0.1) / 0.3
        self.seq_len = tf.placeholder(tf.int64, [None], name='seq_len')
        self.targets = tf.sparse_placeholder(tf.int32, name='targets')
        cell_fn = tf.contrib.rnn.GRUCell
        additional_cell_args = {}
        rnn_fw_1 = cell_fn(num_hidden_1, **additional_cell_args)
        rnn_bw_1 = cell_fn(num_hidden_1, **additional_cell_args)
        rnn_fw_2 = cell_fn(num_hidden_2, **additional_cell_args)
        rnn_bw_2 = cell_fn(num_hidden_2, **additional_cell_args)
        rnn_fw_3 = cell_fn(num_hidden_3, **additional_cell_args)
        rnn_bw_3 = cell_fn(num_hidden_3, **additional_cell_args)
        rnn_fw_4 = cell_fn(num_hidden_4, **additional_cell_args)
        rnn_bw_4 = cell_fn(num_hidden_4, **additional_cell_args)
        with tf.variable_scope('layer1') as (vs1):
            outputs_1, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_1, rnn_bw_1, inputs, self.seq_len, dtype=tf.float32, parallel_iterations=1)
            outputs_1 = tf.concat(axis=2, values=outputs_1)
        with tf.variable_scope('layer2') as (vs2):
            outputs_2, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_2, rnn_bw_2, outputs_1, self.seq_len, dtype=tf.float32, parallel_iterations=1)
            outputs_2 = tf.concat(axis=2, values=outputs_2)
        with tf.variable_scope('layer3') as (vs3):
            outputs_3, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_3, rnn_bw_3, outputs_2, self.seq_len, dtype=tf.float32, parallel_iterations=1)
            outputs_3 = tf.concat(axis=2, values=outputs_3)
        with tf.variable_scope('layer4') as (vs4):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_4, rnn_bw_4, outputs_3, self.seq_len, dtype=tf.float32, parallel_iterations=1)
            outputs = tf.concat(axis=2, values=outputs)
        shape = tf.shape(inputs)
        batch_s, max_timesteps = shape[0], shape[1]
        outputs = tf.reshape(outputs, [-1, num_hidden_4 * 2])
        W = tf.Variable(tf.truncated_normal([num_hidden_4 * 2, num_classes], stddev=0.01), name='ctc_weights')
        b = tf.Variable(tf.constant(0.0, shape=[num_classes]), name='ctc_bias')
        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [batch_s, -1, num_classes])
        logits = tf.transpose(logits, (1, 0, 2))
        self.decoded, log_prob = ctc.ctc_beam_search_decoder(logits, tf.cast(self.seq_len, dtype='int32'))
        self.err = tf.reduce_sum(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets, normalize=False))
        self.saver = tf.train.Saver(max_to_keep=0)
        return


def init_model(session, gpu_id=-1):
    if gpu_id == -1:
        xpu = '/cpu:0'
    else:
        xpu = '/gpu:' + str(gpu_id)
    with tf.device(xpu):
        model = Model()
    tf.global_variables_initializer().run()
    model_dir = './ocr/model/'
    names = os.listdir(model_dir)
    model_file = 'model.ckpt'
    model.saver.restore(session, model_dir + model_file)
    return model


def recog(img_path, model, session):
    dict = './ocr/look_up.txt'
    f = codecs.open(dict, 'r', 'utf-8')
    look_up_lines = f.readlines()
    f.close()
    image = preprocess(img_path)
    img, len = img2input(image)
    feed = {model.inputs0: img, model.seq_len: len}
    decoded_Array = session.run(model.decoded[0], feed_dict=feed)
    decoded_str = get_row(decoded_Array, 0)
    lats = []
    for cnt in decoded_str:
        look_up_line = look_up_lines[cnt]
        lat = look_up_line.split('*')[1]
        lats.append(lat)

    return lats


if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as (session):
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        f = open(input_file, 'r')
        img_names = f.readlines()
        f.close()
        fw = codecs.open(output_file, 'w', 'utf-8')
        #fw.write('#!MLF!#\r\n')
        model = init_model(session, gpu_id=0)
        for filename in img_names:
            filename = filename.strip()
            print(filename)
            if filename.endswith('.png'):
                #raise AssertionError
                dirs = filename.split('/')
                dir = dirs[-2]
                iname = dirs[-1]
		#print(iname)
                rec_path = filename.replace(dir, 'outputRec').replace('png', 'rec')
                fw.write('"' + rec_path + '"\r\n')
                arabic = recog(filename, model, session)
                for ar in arabic:
                    #fw.write(lat + '\r\n')
		    #print(ar)
                    fw.write(ar + " ")

                fw.write('.\r\n')

        fw.close()
    import result


