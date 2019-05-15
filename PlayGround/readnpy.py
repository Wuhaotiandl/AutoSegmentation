import numpy as np
from core.shelper import ShowImage, ExtractInfo

def getTestSet():
    path = r'G:\data\rectum\origin_data\data.npy'
    path_2 = r'G:\data\rectum\origin_data\label.npy'
    img = np.load(path)
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

getTestSet()