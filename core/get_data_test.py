import os
import glob
import numpy as np
import pydicom
import re
import collections
from skimage import measure
import sys


def find_roi_id(rois, roi_pattern):
    for pattern in roi_pattern:
        for i, roi in enumerate(rois):
            if re.match(pattern, roi, re.I):
                return i
    return -1


data_path = r"E:\cervial_P"
roi_pattern = ["bladder"]
save_path = r"F:\nouse\data"
patients_list = os.listdir(data_path)
for patient in patients_list:
    patient_path = os.path.join(data_path, patient)
    # 提取CT图像，并调整顺序与维度
    CT_files = glob.glob(os.path.join(patient_path, "CT*.dcm"))
    slices = [pydicom.dcmread(s, force=True) for s in CT_files]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=False)
    # nn = [s.ImagePositionPatient for s in slices]
    # print(nn)
    # print(slices[0].ImagePositionPatient[2])
    # print(slices[-1].ImagePositionPatient[2])
    if len(slices) > 1:
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    elif len(slices) == 1:
        slice_thickness = slices[0].SliceThickness
    else:
        sys.exit(1)
    SOPInstanceUIDs = [x.SOPInstanceUID for x in slices]
    start_point = np.array(slices[0].ImagePositionPatient)
    print(start_point)

    Space =(slices[0].PixelSpacing)
    temp_spacing = []
    for i in range(len(Space)):
        temp_spacing.append(float(Space[i]))
    spacing = map(float, ([slice_thickness] + temp_spacing))
    spacing = np.array(list(spacing))#分辨率和空间层厚
    print(spacing)

    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    print(np.max(image))#22334
    image = image.astype(np.int16)
    image[image < -1500] = -1024
    image[image > 2976] = 2976
    image_original = image.astype(np.float32)
    print(np.max(image_original))  # 22334
    print(image_original.shape)
    masks = np.zeros(image_original.shape,np.int)
    print(masks.shape)
    print(np.max(masks))

    RS_files = glob.glob(os.path.join(patient_path, "RS*.dcm"))
    RS_meta = pydicom.dcmread(RS_files[0])
    rois = [roi.ROIName for roi in RS_meta.StructureSetROISequence]
    id = find_roi_id(rois, roi_pattern)
    if id:
        Data = []
        for contour in RS_meta.ROIContourSequence[id].ContourSequence:
            contour_data = contour.ContourData
            Data = np.append(Data, contour_data)
        print(len(Data))
        Data = np.array(Data).reshape(-1, 3)
        print(Data.shape)
        X = (np.round((Data[:, 0] - start_point[0]) / spacing[1]).astype(np.int))
        print(X.shape)
        Y = (np.round((Data[:, 1] - start_point[1]) / spacing[2]).astype(np.int))
        print(Y.shape)
        Z = (np.round((Data[:, 2] - start_point[2]) / spacing[0]).astype(np.int))
        print(Z.shape)
        print(np.max(Z))
        masks[Z,X,Y] = 1
        print(np.sum(np.sum(np.sum(masks))))
        save_dir = os.path.join(save_path, patient)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, "images"), image_original)
        np.save(os.path.join(save_dir, "masks"), masks)
    else:
        print(patient)
        print(rois)
        continue