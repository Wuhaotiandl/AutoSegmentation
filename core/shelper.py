import numpy as np
import matplotlib.pyplot as plt
import math
import SimpleITK as itk
from skimage.segmentation import slic
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.measure import regionprops
from skimage.filters import gaussian
import cv2

"""
    this file describes the operation of dcm data
"""

def ReadDcmByItk(path):
    """
        read single dcm data by itk
        ----------------
        Parameters:
            path: path of dcm or nii or nrrd
        ----------------
        Return:
            image_array of dcm data. it's a numpy matrix
    """
    img = itk.ReadImage(path)
    img_array = np.squeeze(itk.GetArrayFromImage(img))
    return img_array

def ResampleDcm(path, outputsapcing= [1, 1, 1]):
    """
        resample the dcm data by itk.
        eg. if the dcm shape is [75, 512, 512]  it can resample the data to [375, 360, 360]  depends on outputspacing
        -------------
        Parameters:
            path: path of dcm, usually it is the path of nii or nrrd
            outputspacing: new resolution space. units of mm
        -------------
        Return:
            the resampled image arrray
    """
    if not path:
        assert 'not found nii files'
    datareader = itk.ImageFileReader()
    datareader.SetFileName(path)
    reader = datareader.Execute()
    """
    t_image = itk.GetArrayFromImage(reader)
    """
    spacing = np.array(reader.GetSpacing())
    size = np.array(reader.GetSize())
    outputsize = np.int16((spacing * size)/np.array(outputsapcing)).tolist()
    miaorgin = reader.GetOrigin()
    direction = reader.GetDirection()
    resample = itk.ResampleImageFilter()
    resample.SetOutputSpacing(outputsapcing)
    resample.SetSize(outputsize)
    resample.SetOutputOrigin(miaorgin)
    resample.SetOutputDirection(direction)
    reader = resample.Execute(reader)

    image_array = itk.GetArrayFromImage(reader)
    return image_array

def cutliver(img):
    """
        cut liver img 106, 267, 292, 65
    """
    return img[80:310, 40:310]

def cutheart(img):
    return img[100:300, 200:380]

def ExtractInfo(imgs, labels):
    """
        从原图中提取出包含肝脏点的图片
        ----------------------
        Parameters:
            imgs: shape为 [number, w, h]  原图
            labels: shape为 [number, w, h] 标签
        ----------------------
        Return:
            new_imgs: shape为 [number, w, h] 提取的图片
            new_labels: shape为 [number, w, h] 提取的标签
    """
    labels_index = []
    new_imgs = []
    new_labels = []
    for i in range(labels.shape[0]):
        varifyImg = labels[i, :, :]
        numbers = sum(varifyImg[varifyImg == 1])
        # 这部分逻辑写的有问题 -- 改
        # numbers = sum(varifyImg[varifyImg == 1])
        if numbers > 0 :
            labels_index.append(i)
    # ShowImage(1, labels[230, :,:], labels[281,:,:], labels[282,:,:])
    for j in labels_index:
        temp_img = np.expand_dims(imgs[j,:,:], 0)
        temp_label = np.expand_dims(labels[j,:,:], 0)
        new_imgs.append(temp_img)
        new_labels.append(temp_label)
    try:
        new_imgs = np.concatenate(([_li for _li in new_imgs]), axis=0)
        new_labels = np.concatenate(([_slice for _slice in new_labels]), axis=0)
        print("提取完成，当前的个数为", str(labels.shape[0]), " 提取个数为", str(new_labels.shape[0]))
    except:
        print("当前出现异常")

    return new_imgs, new_labels

def DataNormalize(dataarray):
    """
        normalize single img.
        (img - min)  /  (max - min + factor)   to ensure the pixel data is range from (0 , 1)* 255
        ------------
        Parameters:
                img_array:  array of image, it's better for one single img
                factor: correction factor
        ------------
        Return:
                new array
    """
    Minpixel = min(dataarray.ravel())
    Maxpixel = max(dataarray.ravel())
    factor = 0.00001
    PatchNorm = np.divide(((dataarray.ravel() - Minpixel) * 255), ((Maxpixel - Minpixel) + factor))
    return PatchNorm.reshape(dataarray.shape)

def PatchExtract_for_eval(region, imgslice):
    count = 0
    img = imgslice
    total=len(region)
    patch_data = []
    patch_coord = []
    region_index = []
    for r in range(len(region)):
        if (region[r].mean_intensity) > 1.5 and (region[r].max_intensity) > 0:
            ymin = int(region[r].centroid[0]) - 16
            ymax = int(region[r].centroid[0]) + 16
            xmin = int(region[r].centroid[1]) - 16
            xmax = int(region[r].centroid[1]) + 16
            if ymin >= 0 and xmin >= 0 and ymax < img.shape[0] and xmax < img.shape[1]:
                pa = img[ymin: ymax, xmin: xmax]
                # [1, 32, 32]
                # pa = np.expand_dims(pa, 0)

                patch_data.append(DataNormalize(pa))
                # 存放边界位置
                patch_coord.append([ymin, ymax, xmin, xmax])
                count += 1
                region_index.append(r)
    return patch_data, patch_coord, region_index

def PatchExtract(region, imgslice, labelslice):
    """
    共用
    :param region:
    :param imgslice:
    :param labelslice:
    :return:
    """
    count = 0
    img = imgslice
    total=len(region)
    labelvalue=[]
    patch_data = []
    patch_coord = []
    patch_liver_index = []
    region_index = []
    for r in range(len(region)):
        if (region[r].mean_intensity)>1.5 and (region[r].max_intensity)>0:
            ymin=int(region[r].centroid[0]) - 16
            ymax=int(region[r].centroid[0]) + 16
            xmin=int(region[r].centroid[1]) - 16
            xmax=int(region[r].centroid[1]) + 16
            if ymin >= 0 and xmin >= 0 and ymax < img.shape[0] and xmax < img.shape[1]:
                flaglabel=(labelslice[ymin:ymax, xmin:xmax] == 1).sum()/labelslice[ymin:ymax, xmin:xmax].size
                if flaglabel == 1:
                    label = 1
                    patch_liver_index.append(r)
                elif flaglabel>=0.5 and flaglabel<1:
                    label = 2
                    patch_liver_index.append(r)
                else:
                    label = 0
                # 存放单个patch [32, 32]
                pa = img[ymin: ymax, xmin: xmax]
                # [1, 32, 32]
                # pa = np.expand_dims(pa, 0)
                patch_data.append(DataNormalize(pa))
                # 存放边界位置
                patch_coord.append([ymin,ymax,xmin,xmax])
                count += 1
                # 存放对应的标签
                labelvalue.append(label)
                region_index.append(r)
#     print('The %d@%d superpixel region patch have done' % (count, total))
    return labelvalue, patch_data, patch_coord, count, region_index, patch_liver_index

def ShowImage(rows=1, *args):
    """
        show the image by plt.
        --------------
        Parameters:
            rows: row of plot, you should give the rows and it will measure the column.
            *args: the data of image, it should be the format of numpy.
        --------------
        Return:
            null
    """
    column = math.ceil(len(args) / rows)
    fig = plt.figure()
    for i in range(rows):
         for j in range(column):
            index = column*i + (j+1)
            plt.subplot(rows, column, index)
            if index <= len(args):
                plt.title("Image:" + str(index))
                plt.imshow(args[index-1])
    plt.show()

def ShowImageList(imglist, rows=1):
    """
        show the image by plt.
        --------------
        Parameters:
            rows: row of plot, you should give the rows and it will measure the column.
            *args: the data of image, it should be the format of numpy.
        --------------
        Return:
            null
    """
    column = math.ceil(len(imglist) / rows)
    fig = plt.figure()
    for i in range(rows):
         for j in range(column):
            index = column*i + (j+1)
            plt.subplot(rows, column, index)
            if index <= len(imglist):
                plt.title("Image:" + str(index))
                plt.imshow(imglist[index-1])
    plt.show()

def normalization_1(img_array, factor=0.0001):
    """
        normalize single img.
        (img - min)  /  (max - min + factor)   to ensure the pixel data is range from (0 , 1)
        ------------
        Parameters:
                img_array:  array of image, it's better for one single img
                factor: correction factor
        ------------
        Return:
                new array
    """
    min_array = min(img_array.ravel())
    max_array = max(img_array.ravel())
    new_img_array = np.divide(((img_array.ravel() - min_array)), ((max_array - min_array) + factor))
    new_img_array = new_img_array.reshape(img_array.shape)
    return new_img_array

def ResizeChannel(img_array):
    """
        turn single channel image into three channels.
        eg.  a = [[[1],[2],[3]]]  --> [[[1,1,1], [2,2,2], [3,3,3]]]
        -------------
        Parameters:
                img_array: array of image or other array, but you must ensure that it's three-dimensional,
                          it's channel must be 1 in three-dimensional
        -------------
        Return:
                shape of new array is [w, h, 3]
    """
    temp_img_array = np.dstack((img_array, img_array))
    new_img_array = np.dstack((temp_img_array, img_array))
    return new_img_array

def SuperpixelExtract_for_rectum(o_img, n_segments):
    img = o_img
    # 高斯去噪
    PatchN = gaussian(img, sigma=0.5)
    # 拓展通道
    slice_colors = ResizeChannel(PatchN)
    superpixel = slic(slice_colors, n_segments, compactness=5, sigma=1)
    slice_entro = entropy(PatchN, disk(5))
    regions = regionprops(superpixel, slice_entro)
    return PatchN, regions, superpixel, slice_colors


def SuperpixelExtract(o_img, n_segments, is_data_from_nii = 1):
    if is_data_from_nii == 1:
        o_img[o_img > 200] = 200
        o_img[o_img < -200] = -200
    else:
        # o_img原本通过阈值分割 在这里已经改变了如果有下面这一步的话，就会导致一个问题，o_img不会改变原来的值，而在提取patch后，它的值还是阈值分割之前的值
        # o_img = o_img.astype(np.int16)
        o_img += np.int16(-1024)
        o_img[o_img < -1500] = -1024
        o_img[o_img > 2976] = 2976
        o_img[o_img > 200] = 200
        o_img[o_img < -150] = -150
    img = o_img
    # ShowImage(1, img)
    PatchN = normalization_1(img)
    # edit on 4/22 归到[0, 1]还是[0, 255]之间？ 归到0~255 之间  归到0 ~ 1 之间
    # PatchN = DataNormalize(img)
    # 高斯去噪
    PatchN = gaussian(PatchN, sigma=0.5)
    # 拓展通道
    slice_colors = ResizeChannel(PatchN)
    superpixel = slic(slice_colors, n_segments, compactness=5, sigma=1)
    slice_entro = entropy(PatchN, disk(5))
    regions = regionprops(superpixel, slice_entro)
    return PatchN, regions, superpixel, slice_colors

def Fill_holes(img):
    im_th = img.copy()
    im_th = im_th.astype(np.uint8)
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_th, mask,  (0, 0), 255)
    rel = im_th.copy()
    a = sum(sum(rel == 255))
    b = sum(sum(rel == 1))
    c = sum(sum(rel == 0))
    #
    rel = rel.astype(np.int64)
    # 注意先后顺序
    rel[rel != 255] = 1
    rel[rel == 255] = 0
    # ShowImage(1, img, im_th, rel)
    return rel

def extract_values_coords(img, value=1):
    """
        提取图中某个值的坐标集合
        ------------------
        Parameters:
            img: 图片，它是一个numpy矩阵
            value: 对应的值
        ------------------
        Returns:
            coords: 返回类型是一个list, list[[h1, w1]，...[hn, wn]], list里面的元素分别是行列对应的值
    """
    coords = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == value:
                coords.append([i, j])
    return coords

def draw_coords_img(img, coords, value=1):
    """
        在img上描边， 不影响原图
        ---------------------
        Parameters:
            img: 原图，它是一个numpy矩阵
            coords: 坐标类型是一个list[[h1, w1],..., [hn, wn]]
        ---------------------
        Returns:
            _img: img的备份，描边之后的图
    """
    _img = img.copy()
    for coord in coords:
        _img[int(coord[0]), int(coord[1])] = value
    return _img

def extract_counters(mask, value=1):
    _mask = mask.copy()
    _mask = mask.astype(np.uint8)
    thresh, contours, hierarchy = cv2.findContours(_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    template = np.zeros_like(_mask)
    cv2.drawContours(template, contours, -1, 1, 2)
    return extract_values_coords(template, value=value)

def find_counters_by(mask, value=1):
    coords = []
    for i in range(mask.shape[0] -1):
        for j in range(mask.shape[1] -1):
            sums = 0
            if i > 0 and j > 0 :
                 sums = mask[i-1, j-1] + mask[i, j-1] + mask[i+1, j-1] + mask[i-1, j] + mask[i, j] + mask[i+1, j] + mask[i-1, j+1] + mask[i+1, j+1]
                 if sums > value and mask[i, j] == value and sums < 8*value:
                     coords.append([i, j])
    return coords

def extract_shuffle_array(start, end, size, data):
    """
        从data里面随机抽取size长度的数据数据
        -----------------------------------
        Parameters:
            start: 开始下标，一般为0
            end: 结束下表，一般为data.shape[0]
            size: 提取长度
            data: 数据，它为np.array矩阵，数据格式为[Number, ] 第一维是number
        ----------------------------------
        Returns:
            返回data里面随机的一批数据
    """
    shuffle = np.random.randint(start, end, size)
    return data[shuffle]
