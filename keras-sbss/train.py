import numpy as np
import glob
from skimage.filters import gaussian
from core.shelper import *
from skimage.segmentation import slic
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.measure import regionprops
import os
import SimpleITK as itk
import logging
from model import sbss_net
from keras.utils import np_utils


def main(seg_path, img_path, n_segments):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    seg_path_list = glob.glob(seg_path)
    record_path = r"G:\Company\record"
    temp_record = r""
    img_path_list = glob.glob(img_path)
    seg_path_list.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0].split('-')[-1]), reverse=False)
    img_path_list.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0].split('-')[-1]), reverse=False)
    #seg_path_list = [r'G:\data\part_LITS17\segmentations\segmentation-0.nii']
    #img_path_list = [ r'G:\data\part_LITS17\volumes\volume-0.nii']
    train_imgs = []
    train_label = []
    for i in range(len(seg_path_list)):
        print("在操作", str(img_path_list[i]))
        # 重采样之后
        imgs = ResampleDcm(img_path_list[i])
        labels = ResampleDcm(seg_path_list[i])
        # 判断当前的数据是否有肝脏数据，如果有则加入，没有则剔除
        imgs, labels = ExtractInfo(imgs, labels)
        for j in range(imgs.shape[0]):
            cut_img = imgs[j, :, :][90:300, 50:300]
            cut_label = labels[j, :, :][90:300, 50:300]
            PatchN, regions, superpixel, slice_colors = SuperpixelExtract(cut_img, n_segments)
            labelvalue, patch_data, patch_coord, count, a1, a2 = PatchExtract(regions, cut_img, cut_label)
            print("handle: "+ str(j))
            if len(patch_data) > 0:
                patch_data = np.concatenate(([_slice for _slice in patch_data]), axis=0)
                train_imgs.append(patch_data)
            if len(labelvalue) > 0:
                train_label.append(labelvalue)
            # model = sbss_net()
            # model.fit(train_imgs, train_label, validation_split=0.3, epochs=10, batch_size=2)
        if i >= 10:
            break
    train_imgs = np.concatenate(([_slice for _slice in train_imgs]), axis=0)
    train_label = np.concatenate(([_slice for _slice in train_label]), axis=0)
    train_label = np_utils.to_categorical(train_label, num_classes=3)
    model = sbss_net()
    model.fit(train_imgs, train_label, validation_split=0.3, epochs=5, batch_size=64, shuffle=True)
    model.save('segliver.h5')
    model_json = model.to_json()
    path = 'sbss.json'
    with open(path, 'w') as json_file:
        json_file.write(model_json)





if __name__ == '__main__':
    seg_path = r'G:\data\part_LITS17\segmentations\*.nii'
    img_path = r'G:\data\part_LITS17\volumes\*.nii'
    n_segments = 810
    main(seg_path, img_path, n_segments)


