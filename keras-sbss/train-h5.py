from model_3cnn import sbss_net
import h5py
from keras.utils import np_utils
import numpy as np
"""
    该文件的训练数据从h5里面提取
"""

def train(data_path, model_save_path):
    train_img = []
    train_label = []
    with h5py.File(data_path) as f:
        train_img = f['Patch'][:]
        train_label = f['Mask'][:]
    model = sbss_net()
    train_img = train_img.astype(np.float32)
    train_img = np.expand_dims(train_img, 3)
    train_label = np_utils.to_categorical(train_label, num_classes=3)
    model.fit(train_img, train_label, validation_split=0.3, epochs=10, batch_size=64, shuffle=True)
    model.save(model_save_path)
    model_json = model.to_json()
    path = 'sbss.json'
    with open(path, 'w') as json_file:
        json_file.write(model_json)



if __name__ == '__main__':
    data_path = r'E:\MyProjects-data-model\data\80patients.h5'
    model_save_path = r'E:\MyProjects-data-model\model\segliver_model_3cnn_10ecrossentry.h5'
    train(data_path, model_save_path)
