import numpy as np
import glob
from core.shelper import *
import logging
from model import sbss_net
from keras.utils import np_utils
import h5py
import os


def main(data_path, n_segments, data_save_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    data_path_folder_names = os.listdir(data_path)
    flag = 0
    # 多少个病人
    for folder_name in data_path_folder_names:
        print("在操作",  folder_name)
        train_imgs = []
        train_label = []
        temp_img_path = os.path.join(data_path, folder_name, "ct_data.npy")
        temp_mask_path = os.path.join(data_path, folder_name, "Heart.npy")
        if os.path.exists(temp_mask_path) and os.path.exists(temp_img_path):
            imgs = np.load(temp_img_path)
            labels = np.load(temp_mask_path)
            # TODO 这步可以更改，可以将分类结果的图像纳入到当前的训练集中
            imgs, labels = ExtractInfo(imgs, labels)
            for j in range(imgs.shape[0]):
                cut_img = imgs[j, :, :][100:300, 200:380]
                cut_label = labels[j, :, :][100:300, 200:380]
                PatchN, regions, superpixel, slice_colors = SuperpixelExtract(cut_img, n_segments, is_data_from_nii=0)
                labelvalue, patch_data, patch_coord, count,  region_index, patch_liver_index = PatchExtract_rot(regions, cut_img, cut_label)
                print("handle: {}, {}".format(str(j), str(len(patch_liver_index))))
                # 查看提取的patch_liver_index是否符合标准，重新生成超像素显示
                # y_shape, x_shape = cut_label.shape[0], cut_label.shape[1]
                # whiteboard_region_2 = np.zeros((y_shape, x_shape))
                # for lindex3 in patch_liver_index:
                #     temp_region = regions[lindex3]
                #     for value in temp_region.coords:
                #         whiteboard_region_2[value[0], value[1]] = 1
                # ShowImage(1, cut_img, cut_label,  whiteboard_region_2)
                if len(patch_data) > 0:
                    patch_data = np.stack(([_slice for _slice in patch_data]), axis=0)
                    train_imgs.append(patch_data)
                if len(labelvalue) > 0:
                    train_label.append(labelvalue)
            ## 此处数据类型时float64
            train_imgs = np.concatenate(([_slice for _slice in train_imgs]), axis=0)
            train_label = np.concatenate(([_slice for _slice in train_label]), axis=0)
            train_imgs = train_imgs.astype(np.float32)
            train_label = train_label.astype(np.int16)
            with h5py.File(os.path.join(data_save_path, str(flag) + '.h5'), 'w') as fwrite:
                fwrite.create_dataset('Patch', data=train_imgs)
                fwrite.create_dataset('Mask', data=train_label)
                print('start storing... ')
        else:
            print("当前的{}数据缺失！".format(folder_name))
        flag += 1
        print("当前处理到{}".format(str(flag)))




if __name__ == '__main__':
    data_path = r'G:\data\heart_data\heart_masks\20190313_heart_masks\masks'
    data_save_path = r'G:\data\heart_data\heart_masks\20190313_heart_masks\78patients_2000_cut'
    n_segments = 2000
    main(data_path, n_segments, data_save_path)


