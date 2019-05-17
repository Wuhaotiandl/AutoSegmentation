from model_3cnn import sbss_net
import h5py
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
import time
import gc
"""
    对于Python内存机制的初步认识
"""
def sty_memory(data_path):
    # 读取文件并不耗费多少内存 为119M
    with h5py.File(data_path) as f:
        # 此时内存占用为4187.4M，train_img里面的类型为np.float64 每一位占用8bit
        train_img = f['Patch'][:]
        # 此时内存占用4181.4M  Mask类型为int32
        train_label = f['Mask'][:]
    # 此时内存占用6215.6M 约涨了2000M, 因为从float64 --> float32 train_img的空间占用缩小一半
    # 但是可以观察出一个问题，原先的train_img内存并未释放,对于train_img而言它占据了整整6个G的空间(4个G float64, 2个G float32)
    train_img_1 = train_img.astype(np.float32)
    # 删除对于train_img的引用计数
    del train_img
    # 回收内存， 此时内存降低到2145.8M
    gc.collect()

# 初始内存占用1.9M  注意文件大小为4167268kb 约莫4030M
sty_memory(r'G:\data\rectum\train_data\train_rectum.h5')