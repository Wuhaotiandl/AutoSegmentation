from core.shelper import *
import numpy as np
import os
import h5py
from segliver_3d_8 import seg_liver3d_8

def extract_data(data_path):
    """
        从多个文件中提取出3d的数据
    """
    train_img = []
    train_label = []
    file_names = os.listdir(data_path)
    file_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for file_name in file_names[:20]:
        file_path = os.path.join(data_path, file_name)
        with h5py.File(file_path) as f:
            train_img.append(f['images'][:])
            train_label.append(f['masks'][:])
            print("{}加载完成".format(file_name))
    train_label = np.concatenate([s for s in train_label], axis=0)
    train_img = np.concatenate([l for l in train_img], axis=0)
    train_label = np.expand_dims(train_label, 4)
    train_img = np.expand_dims(train_img, 4)
    return train_img, train_label

def train(data_path, save_path):
    train_img, train_label = extract_data(data_path)
    model = seg_liver3d_8()
    model.fit(train_img, train_label, validation_split=0.3, epochs=10, batch_size=2, shuffle=True)
    model.save(save_path)
    model_json = model.to_json()
    with open(save_path, 'w') as json_file:
        json_file.write(model_json)

data_path = r'G:\data\liver_data\liver3d_data\8size'
save_path = r'G:\model-store\liver-model\liver_3d'
train(data_path, save_path)