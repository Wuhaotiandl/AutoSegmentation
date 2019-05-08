import numpy as np
from core.shelper import ShowImage, ExtractInfo

def getTestSet():
    path = r'G:\Company\masks\FANG CHEN XIU\ct_data.npy'
    path_2 = r'G:\Company\masks\FANG CHEN XIU\Heart.npy'
    img = np.load(path)
    mask = np.load(path_2)
    img_1, mask_1 = ExtractInfo(img, mask)
    return img_1, mask_1
