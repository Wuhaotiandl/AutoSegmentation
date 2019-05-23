import os
from keras import models
import keras
import sys

def main():
    """
        生成对应的模型图，显示模型结构
    """
    weights_path = r'E:\PycharmProjects\SBSS-CNN\new_program\prog\model'
    json_path = r'seg_liver3D_8.json'
    h5_path = r'seg_liver3D_8.h5'
    print(weights_path)
    with open(os.path.join(weights_path, json_path)) as file:
        _net = models.model_from_json(file.read())
    _net.load_weights(os.path.join(weights_path, h5_path))
    _net.summary()
    keras.utils.plot_model(_net, to_file='seg_liver3d_8.png')

if __name__=='__main__':
    main()
