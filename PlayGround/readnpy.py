import numpy as np
from core.shelper import *

def getTestSet():
    path = r'G:\data\heart_data\val_data\ZHANG GUO LIANG\ct_data.npy'
    path_2 = r'G:\data\heart_data\val_data\ZHANG GUO LIANG\heart.npy'
    img = np.load(path)
    img += np.int16(-1024)
    img[img < -1500] = -1024
    img[img > 2976] = 2976
    img[img < -150] = -150
    img[img > 200] = 200
    mask = np.load(path_2)
    img_1, mask_1 = ExtractInfo(img, mask)
    for i in range(img_1.shape[0]):
        show_img = img_1[i]
        show_mask = mask_1[i]
        show_mask = show_mask.reshape(show_mask.shape[0], show_mask.shape[1])
        show_img = show_img.reshape(show_img.shape[0], show_img.shape[1])
        if i > 60:
            break
        ShowImage(1, show_img, show_mask)
    a = 0

def compare(path_1, path_2):
    image_cyk, SOPInstanceUIDs, ImagePositionPatients, slices, Space, spacing = load_data(path_1)
    image_lxn, SOPInstanceUIDs, ImagePositionPatients, slices, Space, spacing = load_data(path_2)
    image_cyk[image_cyk < -150] = -150
    image_lxn[image_cyk < -150] = -150
getTestSet()