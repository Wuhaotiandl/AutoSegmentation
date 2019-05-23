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
from skimage import morphology
from skimage.measure import label as label_function
import pydicom as dicom
import h5py
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

def resample_dcm_no_change_number(path, outputspacing = [1, 1]):
    """
            在不改变其层厚的情况下，对其进行插值
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
    # 将thickness加进去
    outputspacing.append(spacing[2])
    outputsize = np.int16((spacing * size) / np.array(outputspacing)).tolist()
    miaorgin = reader.GetOrigin()
    direction = reader.GetDirection()
    resample = itk.ResampleImageFilter()
    resample.SetOutputSpacing(outputspacing)
    resample.SetSize(outputsize)
    resample.SetOutputOrigin(miaorgin)
    resample.SetOutputDirection(direction)
    reader = resample.Execute(reader)

    image_array = itk.GetArrayFromImage(reader)
    return image_array


def ResampleDcm(path, outputspacing= [1, 1, 1]):
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
    outputsize = np.int16((spacing * size)/np.array(outputspacing)).tolist()
    miaorgin = reader.GetOrigin()
    direction = reader.GetDirection()
    resample = itk.ResampleImageFilter()
    resample.SetOutputSpacing(outputspacing)
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

def wipe_out_uncenter(regions, x, y):
    """
        心脏 抽出中心点位于身体中心附近的联通区域
    :param regions:
    :param centers:
    :return:
    """
    threshold = 0.3 *x
    center_to_select = []
    for area_property in regions:
        if area_property.centroid[0] > 150 and area_property.centroid[0] < 220 and area_property.centroid[1] > threshold and area_property.centroid[1] < y - threshold:
            center_to_select.append(area_property)

    return center_to_select

def max_label(regions):
    """
        返回最大面积对应的label
    """
    area_list = []
    for i in range(len(regions)):
        area_list.append(regions[i].area)
    v = regions[area_list.index(max(area_list))]
    return v.label

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

def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr

def PatchExtract_rot(region, imgslice, labelslice):
    """
         训练用 左旋90 180 270
        :param region:
        :param imgslice:
        :param labelslice:
        :return:
        """
    count = 0
    img = imgslice
    total = len(region)
    labelvalue = []
    patch_data = []
    patch_coord = []
    patch_liver_index = []
    region_index = []
    for r in range(len(region)):
        if (region[r].mean_intensity) > 1.5 and (region[r].max_intensity) > 0:
            ymin = int(region[r].centroid[0]) - 16
            ymax = int(region[r].centroid[0]) + 16
            xmin = int(region[r].centroid[1]) - 16
            xmax = int(region[r].centroid[1]) + 16
            if ymin >= 0 and xmin >= 0 and ymax < img.shape[0] and xmax < img.shape[1]:
                flaglabel = (labelslice[ymin:ymax, xmin:xmax] == 1).sum() / labelslice[ymin:ymax, xmin:xmax].size
                if flaglabel == 1:
                    label = 1
                    patch_liver_index.append(r)
                elif flaglabel >= 0.5 and flaglabel < 1:
                    label = 2
                    # 90°
                    rot1 = flip90_left(img[ymin: ymax, xmin: xmax])
                    patch_data.append(DataNormalize(rot1))
                    count += 1
                    labelvalue.append(label)
                    # 180°
                    rot2 = flip90_left(rot1)
                    patch_data.append(DataNormalize(rot2))
                    count += 1
                    labelvalue.append(label)
                    # 270°
                    rot3 = flip90_left(rot2)
                    patch_data.append(DataNormalize(rot3))
                    count += 1
                    labelvalue.append(label)

                    patch_liver_index.append(r)
                else:
                    label = 0
                # 存放单个patch [32, 32]
                pa = img[ymin: ymax, xmin: xmax]
                # [1, 32, 32]
                # pa = np.expand_dims(pa, 0)
                patch_data.append(DataNormalize(pa))
                # 存放边界位置
                patch_coord.append([ymin, ymax, xmin, xmax])
                count += 1
                # 存放对应的标签
                labelvalue.append(label)
                region_index.append(r)
    #     print('The %d@%d superpixel region patch have done' % (count, total))
    return labelvalue, patch_data, patch_coord, count, region_index, patch_liver_index

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
        o_img[o_img < -150] = -150
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

def findBodyCenter(one_slice_img):
    x, y = np.shape(one_slice_img)
    x_center = math.ceil(x/2)
    y_center = math.ceil(y/2)
    threshold = 0.3 * x
    binary = one_slice_img > -400
    image = np.array(binary, dtype = np.uint8)
    # 开运算 腐蚀膨胀
    image = morphology.binary_opening(image, selem=morphology.disk(12))
    labeled_image = label_function(image, connectivity=2)
    marked_image = regionprops(labeled_image)
    center_to_select = []
    for area_property in marked_image:
        if area_property.centroid[0]>threshold and area_property.centroid[0]<x-threshold and area_property.centroid[1]>threshold and area_property.centroid[1]<y-threshold:
            center_to_select.append(area_property.centroid)
    if center_to_select:
        distance_all = []
        for one_coordinate in center_to_select:
            distence = np.square(x_center - one_coordinate[0]) + np.square(y_center - one_coordinate[1])
            distance_all.append(distence)
        i = np.argmin(distance_all)
        center = center_to_select[i]
    else:
        center = None
    return center


def cutting(image_size, shift, imgs, pixelx, pixely):
    """
    parameters:
        image_size: Image cut target size
        shift: the absolute of pixelx or pixely
        imgs: Image that needs to be cropped
        pixelx or pixely: The pixel value to move
    """
    z, x, y = np.shape(imgs)
    imgs_new = []
    judge = sum([x > (image_size + shift * 2), y > (image_size + shift * 2)])
    for i, image_std in enumerate(imgs):
        if judge == 2:
            image_std = image_std[int((x - image_size) / 2 + pixelx):int((x + image_size) / 2 + pixelx),
                        int((y - image_size) / 2 + pixely):int((y + image_size) / 2) + pixely]
            imgs_new.append(image_std)
        if judge == 0:
            image_new = np.min(image_std) * np.ones([image_size + shift * 2, image_size + shift * 2], dtype=np.int32)
            image_new[int((image_size + shift * 2 - x) / 2):int((image_size + shift * 2 - x) / 2) + x,
            int((image_size + shift * 2 - y) / 2):int((image_size + shift * 2 - y) / 2) + y] = image_std
            x1, y1 = np.shape(image_new)
            image_std = image_new[int((x1 - image_size) / 2 + pixelx):int((x1 + image_size) / 2 + pixelx),
                        int((y1 - image_size) / 2 + pixely):int((y1 + image_size) / 2) + pixely]
            imgs_new.append(image_std)

        if judge == 1:
            ValueError('the CT image data should be square')
    imgs_new = np.array(imgs_new, np.float32)
    return imgs_new

def recombine_imgs_for_3d(imgs, size=8):
    number, x, y = imgs.shape
    if number > 31:
        z_size = 32
    elif 8 <= number < 32:
        z_size = 8
    else:
        z_size = 4
    block_number = number // z_size
    bias = number % z_size
    z_start_index = block_number * z_size
    z_end_index = number - z_size
    start_imgs = imgs[:z_start_index, :, :]
    end_imgs = imgs[z_end_index:, :, :]
    start_imgs = start_imgs.reshape(block_number, z_size, x, y)
    end_imgs = end_imgs.reshape(1, z_size, x, y)
    rel = np.concatenate(([start_imgs, end_imgs]), axis=0)
    return rel

import os
import sys
from pydicom.filereader import InvalidDicomError

def load_data(data_path):
    #    slices = [dicom.read_file(data_path + os.sep + s, force = True) for s in os.listdir(data_path)]
    # 每个dicom文件的集合
    slices = []
    for s in os.listdir(data_path):
        try:
            single_path = os.path.join(data_path, s)  ## edit by htwu on 2/18
            one_slices = dicom.dcmread(single_path, force=True)
            # print(one_slices)
            # print('****************************')
            # print(one_slices.SOPInstanceUID)
            # sys.exit()
            # one_slices = dicom.read_file(data_path + os.sep +s)
        except IOError:
            print('No such file')
            continue
        except InvalidDicomError:
            print('Invalid Dicom file')
            continue
        slices.append(one_slices)

    slices = [s for s in slices if s.Modality == 'CT']
    # z轴
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=False)
    if len(slices) > 1:
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    elif len(slices) == 1:
        slice_thickness = slices[0].SliceThickness
    else:
        sys.exit(1)

    #    a = slices[0].PatientPosition
    #    b = list(np.array(one_slices.ImageOrientationPatient, np.uint8))
    #    if a != 'HFS':
    #        sys.exit(3)
    #    if b != [1, 0, 0, 0, 1, 0]:
    #        sys.exit(3)

    #    print(slices[0].ImagePositionPatient, slices[0].PixelSpacing)
    ## SOPInstanceUID 用来唯一标识这个图像文件，常由产生这个图像的设备生成
    SOPInstanceUIDs = [x.SOPInstanceUID for x in slices]
    # reshape矩阵
    ImagePositionPatients = np.array([x.ImagePositionPatient for x in slices]).reshape(len(slices), len(
        slices[0].ImagePositionPatient))
    # 病人中各个像素中心点之间的物理距离，每个slice的PixelSpacing均一致
    Space = slices[0].PixelSpacing
    # 这一步是为了将字符串类型转成int
    for s in slices:
        s.SliceThickness = slice_thickness

    temp_spacing = []
    for i in range(len(Space)):
        temp_spacing.append(float(Space[i]))
    spacing = map(float, ([slices[0].SliceThickness] + temp_spacing))
    # spacing[0] 为SliceThickness, spacing[1]为PixelSpacing[0], spacing[1]为PixelSpacing[1]
    spacing = np.array(list(spacing))

    #    spacing[0], spacing[2] = spacing[2], spacing[0]

    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    # 截面
    intercept = slices[0].RescaleIntercept
    # 斜面
    slope = slices[0].RescaleSlope
    # 重新调节斜面
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    # 重新调节截面
    image += np.int16(intercept)
    # 修正
    image[image < -1500] = -1024
    image[image > 2976] = 2976
    image_original = image.astype(np.float32)

    print('image_original.shape： ', image_original.shape)
    print('spacing', spacing)

    return image_original, SOPInstanceUIDs, ImagePositionPatients, slices, Space, spacing

def get_min_max_layer_num(classess_value, slices):
    """
    得到需要预测的层号
    :param classess_value:
    :param slices:
    :return:
    """
    # 心脏在第7类,可能第8类顶层几层也有
    if 7 not in classess_value:
        print('not find the class 7')
        sys.exit(10104202)
    heart_min = classess_value.index(7) - 8
    end_seven = len(classess_value) - classess_value[::-1].index(7) - 1
    if end_seven - classess_value.index(7) < 40:
        heart_max = end_seven
    else:
        heart_max = classess_value.index(7) + 40
    print('classify_7th_layer:', classess_value.index(7), '-', end_seven)

    if heart_min < 0:
        heart_min = 0
    if heart_max >= len(slices):
        heart_max = len(slices) - 1
    print('layer_to_predict: ', heart_min, '-', heart_max)

    if heart_max - heart_min < 8:
        raise ('The CT is too short')

    return heart_min, heart_max

def recombine_data(data_path, img, label):
    """
        将一个文件列表下的多个h5文件合并为一个矩阵
        v2  -- 改进
    """
    file_list = os.listdir(data_path)
    train_image = []
    train_label = []
    flag = 0
    for file_name in file_list:
        with h5py.File(os.path.join(data_path, file_name)) as file:
            train_image.append(file[img][:])
            train_label.append(file[label][:])
            print("加载{}".format(str(flag)))
            flag += 1
    train_image = np.concatenate(([_slice for _slice in train_image]), axis=0)
    train_label = np.concatenate(([_slice for _slice in train_label]), axis=0)
    return train_image, train_label

