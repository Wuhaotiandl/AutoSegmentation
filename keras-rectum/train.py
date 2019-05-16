from model_3cnn import sbss_net
import h5py
from keras.utils import np_utils
import numpy as np
import os

from model_3cnn import sbss_net
import h5py
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
import time

"""
    该文件的训练数据从h5里面提取 
    直肠的数据训练
"""

def train(data_path, model_save_path, model_json_save_path):
    sess = tf.Session()
    with tf.device("/gpu:0"):
        train_img = []
        train_label = []
        with h5py.File(data_path) as f:
            train_img = f['Patch'][:]
            train_label = f['Mask'][:]
        model = sbss_net()
        train_img = train_img.astype(np.float32)
        train_img = np.expand_dims(train_img, 3)
        train_label = np_utils.to_categorical(train_label, num_classes=3)
        start_time = time.time()
        model.fit(train_img, train_label, validation_split=0.3, epochs=10, batch_size=64, shuffle=True)
        end_time = time.time()
        print("耗时：", str(end_time - start_time))
        model.save(model_save_path)
        model_json = model.to_json()
        with open(model_json_save_path, 'w') as json_file:
            json_file.write(model_json)



if __name__ == '__main__':
    # # sess = tf.Session(config=tf.ConfigProto(device_count={'cpu': 0}))
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    data_path = r'G:\data\rectum\train_data\train_rectum.h5'
    model_json_save_path = r'G:\model-store\rectum-model\segRectum_model_3cnn_10epoch_crossentry.json'
    model_save_path = r'G:\model-store\rectum-model\segRectum_model_3cnn_10epoch_crossentry.h5'
    train(data_path, model_save_path, model_json_save_path)
