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
    直肠的验证模型
    注意：数据格式是h5格式的文件 
    事实证明，在超像素预测的时候，h5文件里面存放的是patch，而patch是无法复原回原图的
    所以这里还是采用原数据，从原数据里面随机抽取50张来做测试
"""
def validate(h5_path, data_path):
    origin_image = np.load(data_path)
    origin_image = origin_image.reshape(origin_image.shape[0], origin_image.shape[1], origin_image.shape[2])
    # shuffle_array = np.random.randint(0, origin_image.shape[0], 50)
    #
    # origin_image = origin_image[shuffle_array]

    # origin_image, origin_mask = ExtractInfo(origin_image, origin_mask)
    model = load_model(h5_path)
    _sum = 0
    for i in range(origin_image.shape[0]):
        slice_ = origin_image[i, :, :]
        # ShowImage(1, slice_)
        # ShowImage(2, slice_, label_, cut_slice, cut_label)
        PatchNorm,  region, superpixel, slice_entro = SuperpixelExtract_for_rectum(slice_, 1000)
        patch_data, patch_coord, region_index = PatchExtract_for_eval(region, slice_)
        Patch_test = np.array(patch_data)
        Patch_test = Patch_test.astype(np.float32)
        Patch_test = np.expand_dims(Patch_test, -1)
        prediction = model.predict(Patch_test)
        prediction = np.argmax(prediction, 1)
        y_shape, x_shape = slice_.shape[0], slice_.shape[1]
        whiteboard = np.zeros((y_shape, x_shape))
        whiteboard_region = np.zeros((y_shape, x_shape))
        # whiteboard_region_2 = np.zeros((y_shape, x_shape))
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
            last_finish = morphology.erosion(last_finish, morphology.disk(3))
            # 取contours
            # coords = find_counters_by(last_finish, 1)
            coords = extract_counters(last_finish)
            draw = draw_coords_img(slice_, coords, value=0.9)

            # ShowImage(3, slice_, whiteboard_region,  whiteboard_region_after,
            #           whiteboard_region_after_remove, FillHolesFinish, blurbinary_rel, last_finish, draw)
            ShowImage(2, slice_, draw)
            # dice = 2 * np.sum(slice_*last_finish)/(np.sum(last_finish) + np.sum(slice_))
            # _sum += dice
            # print(dice)
        else:
            ShowImage(1, slice_)
            print("该行无预测")
    # print('平均dice系数为：',  str(_sum / origin_image.shape[0]))
    a = 1

h5_path = r'C:\Users\formy\Documents\Tencent Files\859370476\FileRecv\segRectum_model_3cnn_20epoch_crossentry.h5'
dcm_path = r'G:\data\rectum\test_data\1.2.840.113619.2.55.3.168489271.843.1554297849.997\images.npy'
validate(h5_path, dcm_path)
