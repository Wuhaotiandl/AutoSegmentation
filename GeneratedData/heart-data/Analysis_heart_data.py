import numpy as np
from core.shelper import ShowImage, ExtractInfo
"""
    数据分析
"""
def getTestSet():
    path = r'G:\data\masks\WU ZHENG QING\ct_data.npy'
    path_2 = r'G:\data\masks\WU ZHENG QING\Heart.npy'
    img = np.load(path)
    img = img.astype(np.int16)
    img += np.int16(-1024)
    img[img < -1500] = -1024
    img[img > 2976] = 2976
    mask = np.load(path_2)
    img_1, mask_1 = ExtractInfo(img, mask)
    # ShowImage(1, img[0])
    # for i in range(img.shape[0]):
    #     ShowImage(1, img[i])
    for j in range(img_1.shape[0]):
        ShowImage(1, mask_1[j], mask_1[j][100:300, 200:380])
    return img_1, mask_1

getTestSet()

