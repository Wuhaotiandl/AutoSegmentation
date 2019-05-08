import cv2
import numpy as np
from readnpy import getTestSet
from core.shelper import *
from skimage import morphology, measure
from skimage.measure import grid_points_in_poly, find_contours
"""
    acknowledge: findContours里返回的contours里包含了所有的边界轮廓点， 但是其返回的边界轮廓点很少，并不能完全表示轮廓
    contours 是一个list  里面包含有n个array数组，代表n个轮廓
"""
def ExtractCountours():
    mask_path = r'G:\Company\masks\CHEN SHENG\Heart.npy'
    img, mask = getTestSet()
    img_1 = img[20,:,:]
    t1 = mask[20,:,:]
    t2 = t1.copy()
    t2 = t2.astype(np.uint8)
    mycoords = find_counters_by(t2, 1)
    mytemplate = np.zeros_like(t2)
    myimg = draw_coords_img(mytemplate, mycoords)
    # cv2.imshow(myimg)

    ShowImage(1, myimg)
    # 尝试先二值化图片
    # ret, binary = cv2.threshold(t2, 0.5, 255, cv2.THRESH_BINARY)
    thresh, contours, hierarchy = cv2.findContours(t2, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    template = np.zeros_like(t2)

    # 其中参数2就是得到的contours，参数3表示要绘制哪一条轮廓，-1表示绘制所有轮廓，参数4是颜色（B/G/R通道，所以(0,0,255)表示红色），参数5是线宽
    cv2.drawContours(template, contours, -1, 1, 1)
    # template = morphology.dilation(template, morphology.disk(1))
    # template = morphology.erosion(template, morphology.disk(1))
    coords = extract_values_coords(template, 1)
    img_2 = draw_coords_img(img_1, coords, value=255)
    ShowImage(1,  img_1,  img_2)
    a = 0


ExtractCountours()