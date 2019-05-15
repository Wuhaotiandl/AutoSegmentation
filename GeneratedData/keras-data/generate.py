import numpy as np
import glob
from core.shelper import *
import logging
from model import sbss_net
from keras.utils import np_utils
import h5py


def main(seg_path, img_path, n_segments, data_save_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    seg_path_list = glob.glob(seg_path)
    record_path = r"G:\Company\record"
    temp_record = r""
    img_path_list = glob.glob(img_path)
    seg_path_list.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0].split('-')[-1]), reverse=False)
    img_path_list.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0].split('-')[-1]), reverse=False)
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
            cut_img = imgs[j, :, :][91: 282, 50:307]
            cut_label = labels[j, :, :][91: 282, 50:307]
            PatchN, regions, superpixel, slice_colors = SuperpixelExtract(cut_img, n_segments)
            labelvalue, patch_data, patch_coord, count,  region_index, patch_liver_index = PatchExtract(regions, cut_img, cut_label)
            print("handle: " + str(j))
            if len(patch_data) > 0:
                # patch_data.shape = [Number, 32, 32] 所有的_slice必须具有相同的shape才能用_slice
                patch_data = np.stack(([_slice for _slice in patch_data]), axis=0)
                patch_data = patch_data.astype(np.uint8)
                train_imgs.append(patch_data)
            if len(labelvalue) > 0:
                train_label.append(labelvalue)
            # model = sbss_net()
            # model.fit(train_imgs, train_label, validation_split=0.3, epochs=10, batch_size=2)
        if i >= 50:
            break
    train_imgs = np.concatenate(([_slice for _slice in train_imgs]), axis=0)
    # train_imgs = np.expand_dims(train_imgs, 3)
    train_label = np.concatenate(([_slice for _slice in train_label]), axis=0)

    print('start storing... ')
    # 保存
    with h5py.File(data_save_path, 'w') as fwrite:
        fwrite.create_dataset('Patch', data=train_imgs)
        fwrite.create_dataset('Mask', data=train_label)
        print("Finish All")



if __name__ == '__main__':
    seg_path = r'H:\LiverData\LITS17\segmentations\*.nii'
    img_path = r'H:\LiverData\LITS17\volumes\*.nii'
    data_save_path = r'F:\practice\liver\100patients.h5'
    n_segments = 758
    main(seg_path, img_path, n_segments, data_save_path)


