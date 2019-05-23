import numpy as np
from core.shelper import *
import pydicom as dicom
import SimpleITK as itk
"""
    这个文件主要是研究dcm图像特征
"""
def threshold_segmentation(data_path, is_npy = 1):
    if is_npy == 1:
        imgs = np.load(data_path)
        one_slice_img = imgs[0]
    elif is_npy == 0:
        # 如果是单张
        slices = dicom.dcmread(data_path)
        one_slice_img = slices.pixel_array
        slope = slices.RescaleSlope
        intercept = slices.RescaleIntercept
        # 原始图像类型为 np.uint16
        one_slice_img = one_slice_img.astype(np.int16)
    else:
        slices = ReadDcmByItk(data_path)
        one_slice_img = slices[0]
        one_slice_img[one_slice_img < -150] = -150
        one_slice_img[one_slice_img > 200] = 200
        ShowImage(1, one_slice_img)
    threshold_1 = one_slice_img.copy()
    # 这一步是将图像值转成CT(HU)值
    threshold_1 += np.int16(-1000)
    # <-1500代表其是空气，也可以看作是背景
    threshold_1[threshold_1 < -1500] = -1024
    # 过于大 2976代表是金属伪影
    threshold_1[threshold_1 > 2976] = 2976
    threshold_2 = threshold_1.copy()

    threshold_2[threshold_2 < -150] = -150
    threshold_2[threshold_2 > 100] = 100
    ShowImage(1, one_slice_img, threshold_1, threshold_2)
    threshold_3 = threshold_1.copy()
    # 这一步有两个目的，第一，首先器官的CT值一般在0~200 而脂肪在-80~-120，而骨头一般大于200
    # 所以其可以将器官与骨头还有空气分开， 第二，将CT值缩小至 -150~200的范围，可以使图像成像更为清晰
    threshold_3[threshold_3 < -150] = -150
    threshold_3[threshold_3 > 200] = 200
    ShowImage(1, threshold_3)
    ShowImage(1, one_slice_img, threshold_2, threshold_3)

# data_path = r'G:\data\mixdata\Arts\chen tingmin A1\CT24678.40.dcm'
# data_path = r'G:\data\heart_data\masks\LIN JIAN MIN\ct_data.npy'
data_path = r'H:\LiverData\LITS17\volumes\volume-0.nii'
threshold_segmentation(data_path, is_npy=2)