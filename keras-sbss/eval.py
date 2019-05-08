from core.shelper import *
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras.models import load_model
from model_3cnn_focalloss import focal_loss
from skimage import morphology, measure
from skimage.measure import regionprops
import cv2

"""
    测试下  
"""

def validate(json_path, h5_path, data_path, mask_path=None):
    data_nii = ResampleDcm(data_path)
    mask_nii = ResampleDcm(mask_path)
    segment_number = 810
    data_nii, mask_nii = ExtractInfo(data_nii, mask_nii)
    # model = load_model(h5_path, custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0)})
    model = load_model(h5_path)
    _sum = 0
    for i in range(data_nii.shape[0]):
        # 在原图验证
        slice_ = data_nii[i, :, :]
        label_ = mask_nii[i, :, :]
        cut_slice = cutliver(img=slice_)
        cut_label = cutliver(img=label_)
        # ShowImage(2, slice_, label_, cut_slice, cut_label)
        PatchNorm,  region, superpixel, slice_entro = SuperpixelExtract(cut_slice, segment_number)
        labelvalue, patch_data, patch_coord, count, region_index, patch_liver_index = PatchExtract(region, cut_slice, cut_label)
        Patch_test, Label_test = np.array(patch_data), np.array(labelvalue)
        Patch_test = np.expand_dims(Patch_test, -1)
        Label_test = np_utils.to_categorical(Label_test, num_classes=3)


        # model.load_weights(r'F:\practice\try\weights-improvement-04-0.77.hdf5')
        # Evaluate the model with the metrics defined earlier
        loss, accuracy = model.evaluate(Patch_test, Label_test)
        print("loss: %g,training accuracy: %g" % (loss, accuracy))
        prediction = model.predict(Patch_test)
        prediction = np.argmax(prediction, 1)
        y_shape, x_shape = cut_slice.shape[0], cut_slice.shape[1]
        whiteboard = np.zeros((y_shape, x_shape))
        whiteboard_region = np.zeros((y_shape, x_shape))
        whiteboard_region_2 = np.zeros((y_shape, x_shape))
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
        for lindex3 in patch_liver_index:
            temp_region = region[lindex3]
            for value in temp_region.coords:
                whiteboard_region_2[value[0], value[1]] = 1
        # whiteboard_region_2_after = morphology.dilation(whiteboard_region_2, morphology.disk(1))
        # whiteboard_region_2_after = morphology.erosion(whiteboard_region_2_after, morphology.disk(1))
        # 膨胀腐蚀
        whiteboard_region_after = morphology.dilation(whiteboard_region, morphology.disk(5))
        whiteboard_region_after = morphology.erosion(whiteboard_region_after, morphology.disk(5))
        whiteboard_region_after_remove = measure.label(whiteboard_region_after, connectivity=2)
        afterregions = regionprops(whiteboard_region_after_remove)

        validate_area = []

        for i in range(len(afterregions)):
            validate_area.append(afterregions[i].area)
        if len(validate_area) > 0:
            # 去除外围最小联通区域
            whiteboard_region_after_remove[whiteboard_region_after_remove != validate_area.index(max(validate_area)) + 1] = 0
            whiteboard_region_after_remove[whiteboard_region_after_remove == validate_area.index(max(validate_area)) + 1] = 1
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
            draw = draw_coords_img(cut_slice, coords, value=200)


            ShowImage(3, slice_, label_, whiteboard_region_2, whiteboard_region,  whiteboard_region_after,
                      whiteboard_region_after_remove, FillHolesFinish, blurbinary_rel, last_finish, draw)
            # ShowImage(2, slice_, label_, draw)
            dice = 2 * np.sum(cut_label*last_finish)/(np.sum(last_finish) + np.sum(cut_label))
            _sum += dice
            print(dice)
        else:
            print("该行无预测")
    print('平均dice系数为：',  str(_sum / data_nii.shape[0]))
    a = 1

# 结果dice 系数有点差异
# 0.94, 0.87, 0.89, 0.77, 0.85
# 130：0.7325；125： 0.83
json_path = r'H:\sbss-train-model-store\100patients-50epochs'
# h5_path = r'H:\sbss-train-model-store\100patients-50epochs\segliver_50.h5'
h5_path = r'H:\sbss-train-model-store\100patients-50epochs\segliver_50.h5'
# h5_path = r'H:\sbss-train-model-store\50patients-5epochs-improvemodel-3conv\segliver_2.h5'
data_path = r'H:\LiverData\LITS17\volumes\volume-0.nii'
mask_path = r'H:\LiverData\LITS17\segmentations\segmentation-0.nii'
validate(json_path, h5_path, data_path, mask_path)
