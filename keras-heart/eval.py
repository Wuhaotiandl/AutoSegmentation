#coding=utf-8
from core.shelper import *
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras.models import load_model
from skimage import morphology, measure
from skimage.measure import regionprops
import cv2
import os
import glob
"""
    合并测试
"""
def validate(dcm_path, mask_path, h5_path):
    segment_number = 700
    dcm_files = np.load(dcm_path)
    dcm_masks = np.load(mask_path)
    model = load_model(h5_path)
    dcm_files, dcm_masks = ExtractInfo(dcm_files, dcm_masks)
    for dcm_index,dfile in enumerate(dcm_files):
        cut_file = cutheart(dfile)
        # ShowImage(1, dfile, cut_file)
        PatchNorm, region, superpixel, slice_entro = SuperpixelExtract(cut_file, segment_number, is_data_from_nii=0)

        patch_data, patch_coord, region_index = PatchExtract_for_eval(region, cut_file)
        patch_eval = np.stack(patch_data)
        patch_eval = np.expand_dims(patch_eval, -1)
        # patch_eval = np.array(patch_data)
        prediction = model.predict(patch_eval)
        prediction = np.argmax(prediction, 1)
        expand_y, expand_x = cut_file.shape[0], cut_file.shape[1]
        whiteboard = np.zeros((expand_y, expand_x))
        whiteboard_region = np.zeros((expand_y, expand_x))
        # 存放预测的肝脏为1和2的个数
        liver_index = []

        for index, value in enumerate(prediction):
            if value == 1:
                liver_index.append(index)
            if value == 2:
                liver_index.append(index)
        for lindex in liver_index:
            coord = patch_coord[lindex]
            whiteboard[coord[0]: coord[1], coord[2]: coord[3]] = 1
        for lindex2 in liver_index:
            temp_region = region[region_index[lindex2]]
            for value in temp_region.coords:
                whiteboard_region[value[0], value[1]] = 1
        whiteboard_region_after = morphology.dilation(whiteboard_region, morphology.disk(5))
        whiteboard_region_after = morphology.erosion(whiteboard_region_after, morphology.disk(5))
        whiteboard_region_after_remove = measure.label(whiteboard_region_after, connectivity=2)
        afterregions = regionprops(whiteboard_region_after_remove)

        validate_area = []

        for i in range(len(afterregions)):
            validate_area.append(afterregions[i].area)
        if len(validate_area) > 0:
            # 去除外围最小联通区域
            whiteboard_region_after_remove[
                whiteboard_region_after_remove != validate_area.index(max(validate_area)) + 1] = 0
            whiteboard_region_after_remove[
                whiteboard_region_after_remove == validate_area.index(max(validate_area)) + 1] = 1
            # 泛洪算法fill hole
            FillHolesFinish = Fill_holes(whiteboard_region_after_remove)

            # 二值边界平滑处理
            blurbinary_rel = FillHolesFinish.copy()
            for tag in range(10):
                blurbinary_rel = morphology.dilation(blurbinary_rel, morphology.disk(3))
                blurbinary_rel = morphology.erosion(blurbinary_rel, morphology.disk(2))
            last_finish = morphology.erosion(blurbinary_rel, morphology.disk(4))
            last_finish = morphology.erosion(last_finish, morphology.disk(2))
            # 取contours
            # coords = find_counters_by(last_finish, 1)
            coords = extract_counters(last_finish)
            draw = draw_coords_img(cut_file, coords, value=200)
            ShowImage(2, dfile, dcm_masks[dcm_index], cut_file, whiteboard, whiteboard_region, whiteboard_region_after,draw)

dcm_path = r'G:\data\heart_data\masks\YU SHAN SONG\ct_data.npy'
mask_path = r'G:\data\heart_data\masks\YU SHAN SONG\Heart.npy'
h5_path = r'G:\model-store\heart-model\segliver_model_3cnn_10ecrossentry.h5'
validate(dcm_path, mask_path, h5_path)
