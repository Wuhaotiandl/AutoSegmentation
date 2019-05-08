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
    train_imgs = []
    train_label = []
    for folder_name in data_path_folder_names:
        print("在操作",  folder_name)
        temp_img_path = os.path.join(data_path, folder_name, "ct_data.npy")
        temp_mask_path = os.path.join(data_path, folder_name, "Heart.npy")

        if os.path.exists(temp_mask_path) and os.path.exists(temp_img_path):
            imgs = np.load(temp_img_path)
            labels = np.load(temp_mask_path)
            # 判断当前的数据是否有心脏数据，如果有则加入，没有则剔除
            imgs, labels = ExtractInfo(imgs, labels)
            for j in range(imgs.shape[0]):
                cut_img = imgs[j, :, :][100:300, 200:380]
                cut_label = labels[j, :, :][100:300, 200:380]
                PatchN, regions, superpixel, slice_colors = SuperpixelExtract(cut_img, n_segments, is_data_from_nii=0)
                labelvalue, patch_data, patch_coord, count,  region_index, patch_liver_index = PatchExtract(regions, cut_img, cut_label)
                print("handle: {}, {}".format(str(j), str(len(patch_liver_index))))
                # 查看提取的patch_liver_index是否符合标准，重新生成超像素显示
                # y_shape, x_shape = cut_label.shape[0], cut_label.shape[1]
                # whiteboard_region_2 = np.zeros((y_shape, x_shape))
                # for lindex3 in patch_liver_index:
                #     temp_region = regions[lindex3]
                #     for value in temp_region.coords:
                #         whiteboard_region_2[value[0], value[1]] = 1
                # ShowImage(1, whiteboard_region_2)
                if len(patch_data) > 0:
                    # patch_data.shape = [Number, 32, 32] 所有的_slice必须具有相同的shape才能用_slice
                    patch_data = np.stack(([_slice for _slice in patch_data]), axis=0)
                    train_imgs.append(patch_data)
                if len(labelvalue) > 0:
                    train_label.append(labelvalue)
        else:
            print("当前的{}数据缺失！".format(folder_name))
    train_imgs = np.concatenate(([_slice for _slice in train_imgs]), axis=0)
    train_label = np.concatenate(([_slice for _slice in train_label]), axis=0)
    print('start storing... ')
    # 保存
    with h5py.File(data_save_path, 'w') as fwrite:
        fwrite.create_dataset('Patch', data=train_imgs)
        fwrite.create_dataset('Mask', data=train_label)
        print("Finish All")



if __name__ == '__main__':
    data_path = r'G:\data\heart_data\masks'
    data_save_path = r'G:\data\heart_data\masks\40patients_2000.h5'
    n_segments = 2000
    main(data_path, n_segments, data_save_path)


