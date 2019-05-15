import numpy as np
import glob
from core.shelper import *
import logging
from model import sbss_net
from keras.utils import np_utils
import h5py
"""
    直肠提取数据
"""

def main(seg_path, img_path, n_segments, train_path, val_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    origin_image = np.load(img_path)
    origin_mask = np.load(seg_path)
    origin_mask = origin_mask.reshape(origin_mask.shape[0],  origin_mask.shape[1], origin_mask.shape[2])
    origin_image = origin_image.reshape(origin_image.shape[0], origin_image.shape[1], origin_image.shape[2])
    origin_image, origin_mask = ExtractInfo(origin_image, origin_mask)
    train_imgs = []
    train_label = []
    val_imgs = []
    val_label = []
    for i in range(origin_image.shape[0]):
        # shape 为 [128, 128, 1]
        single_img = origin_image[i]
        single_mask = origin_mask[i]
        # shape 为 [128, 128]
        PatchN, regions, superpixel, slice_colors = SuperpixelExtract_for_rectum(single_img, n_segments)
        labelvalue, patch_data, patch_coord, count,  region_index, patch_liver_index = PatchExtract_expand(regions, single_img, single_mask)
        print("handle: " + str(i))
        if len(patch_data) > 0:
            patch_data = np.stack(([_slice for _slice in patch_data]), axis=0)
            if i < origin_mask.shape[0] / 10 * 7:
                train_imgs.append(patch_data)
                train_label.append(labelvalue)
            else:
                val_imgs.append(patch_data)
                val_label.append(labelvalue)
        # 查看提取的patch_liver_index是否符合标准，重新生成超像素显示
        # y_shape, x_shape = single_img.shape[0], single_img.shape[1]
        # whiteboard_region_2 = np.zeros((y_shape, x_shape))
        # for lindex3 in patch_liver_index:
        #     temp_region = regions[lindex3]
        #     for value in temp_region.coords:
        #         whiteboard_region_2[value[0], value[1]] = 1
        # ShowImage(1, single_img, single_mask, whiteboard_region_2)
    train_imgs = np.concatenate(([_slice for _slice in train_imgs]), axis=0)
    train_label = np.concatenate(([_slice for _slice in train_label]), axis=0)
    val_imgs = np.concatenate(([_slice for _slice in val_imgs]), axis=0)
    val_label = np.concatenate(([_slice for _slice in val_label]), axis=0)
    print('start storing... ')
    # 保存
    with h5py.File(train_path, 'w') as fwrite:
        fwrite.create_dataset('Patch', data=train_imgs)
        fwrite.create_dataset('Mask', data=train_label)
        print("Finish Store train data")
    with h5py.File(val_path, 'w') as fwrite:
        fwrite.create_dataset('Patch', data=val_imgs)
        fwrite.create_dataset('Mask', data=val_label)
        print("Finish Store val data")




if __name__ == '__main__':
    seg_path = r'F:\rectum\label.npy'
    img_path = r'F:\rectum\data.npy'
    train_data_save_path = r'F:\practice\rectum\train_rectum.h5'
    val_data_save_path = r'F:\practice\rectum\val_rectum.h5'
    n_segments = 1000
    main(seg_path, img_path, n_segments, train_data_save_path, val_data_save_path)




