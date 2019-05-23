import numpy as np
from core.shelper import *
import h5py
import os
import glob
"""
    针对nii文件进行提取
    nii文件已经对原始dcm图像进行z轴排序，其读取过后的均为HU值，无需公式转换，此外它的类型为float32类型
"""
def generate(data_path, seg_path, save_path, z_size):
    # TODO Step1 读取nii文件，并插值
    data = resample_dcm_no_change_number(data_path, outputspacing=[1.1, 1.1])
    mask = resample_dcm_no_change_number(seg_path, outputspacing=[1.1, 1.1])
    # TODO Step2 阈值分割，限定当前的HU范围
    data[data < -150] = -150
    data[data > 200] = 200
    mask[mask != 0] = 1  # 可能会有一些值为2的点
    # TODO Step3 根据mask提取含有肝脏的图片
    data, mask = ExtractInfo(data, mask)
    if len(data) > 0:
        # TODO Step4 提取身体中心点
        center_position = findBodyCenter(data[len(data)//2, :, :])
        # TODO Step5 依据身体中心，裁剪出一张256 * 256 的图片
        offsetY = int(center_position[0] - data.shape[1] / 2)  # 行
        offsetX = -30  # 列
        shift = np.max([abs(offsetX), abs(offsetY)])
        imgs = cutting(256, shift, data, offsetY, offsetX)  # shape为 [number, 256, 256]
        labels = cutting(256, shift, mask, offsetY, offsetX)
        # TODO Step6 将imgs组成一个个 [number, z_size, 256, 256, 1]的单元
        train_imgs = recombine_imgs_for_3d(imgs, z_size)
        train_mask = recombine_imgs_for_3d(labels, z_size)
        # TODO Step7 将训练数据存入h5文件中 （train_mask的shape为 [number, size, 256, 256]）
        prefix = data_path.split('\\')[-1].split('.')[0].split('-')[-1]
        with h5py.File(os.path.join(save_path, 'liver3d_8size_' + prefix + '.h5'), 'w') as fwriter:
            fwriter.create_dataset("images", data=train_imgs)
            fwriter.create_dataset("masks",  data=train_mask)
            print("Finish Handle volume-{}".format(prefix))

def generate_all(img_path, seg_path, save_path, size=8):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_list = glob.glob(img_path)
    seg_list = glob.glob(seg_path)
    img_list.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0].split('-')[-1]), reverse=False)
    seg_list.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0].split('-')[-1]), reverse=False)
    for index in range(len(img_list)):
        generate(img_list[index], seg_list[index], save_path, z_size=size)








img_path = r'H:\LiverData\LITS17\volumes\*.nii'
seg_path = r'H:\LiverData\LITS17\segmentations\*.nii'
save_path = r'G:\data\liver_data\liver3d_data\8size'
generate_all(img_path, seg_path, save_path, size=8)