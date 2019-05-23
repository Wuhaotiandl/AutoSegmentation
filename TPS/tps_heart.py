from core.shelper import *
import numpy as np
import json
from keras import models
def  main_predict(data_path, save_path, model_path, path_ofclassess_json,json_name):
    image_original, SOPInstanceUIDs, ImagePositionPatients, slices, Space, spacing = load_data(data_path)
    # 如果当前的分类json文件存在
    if os.path.exists(os.path.join(path_ofclassess_json, 'series_classess.json')):
        information_classess = readClassifyJson(path_ofclassess_json)
        classess_value = [information_classess[i.SOPInstanceUID] for i in slices]
    else:
        classess_value = main_classify(data_path, model_path)
        rst_json_list = []
        for i in range(len(classess_value)):
            temp_dict = {}
            temp_dict["SOPInstanceUID"] = SOPInstanceUIDs[i]
            temp_dict["classess_value"] = str(classess_value[i])
            rst_json_list.append(temp_dict)
        json_file_name = os.path.join(path_ofclassess_json, 'series_classess.json')
        with open(json_file_name, "w") as file:
            json.dump(rst_json_list, file)
    heart_min, heart_max =get_min_max_layer_num(classess_value, image_original)

def readClassifyJson(path_ofclassess_json):
    """
    parameters:
       path_ofclasses_json: the json file path that has the classification information
    function:
       extract the classification information in series_classess.json
    """
    path_ofclassess_json = os.path.join(path_ofclassess_json, 'series_classess.json')
    all_sopinstanceuid = []
    all_classess = []
    with open(path_ofclassess_json, 'r') as fp:
        classfy_result = json.load(fp)
        for one_classfy_result in classfy_result:
            uid = one_classfy_result['SOPInstanceUID']
            all_sopinstanceuid.append(uid)
            value = one_classfy_result['classess_value']
            all_classess.append(int(value))
            information_classess = dict(zip(all_sopinstanceuid, all_classess))
    return information_classess

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

# =============================================================================
# 主分类函数
# path_of_ct是CT的路径，path_of_weights是model所在的路径
# =============================================================================
def main_classify(path_of_ct, path_of_weights):
    # 这是插值之后的图片，这是SOPInstanceUID
    # imgs_data 插值过后图像大小 （499，499，157）
    imgs_data, sop_uid = load_imgs_ct(path_of_ct)
    net = load_net(path_of_weights)
    result = predict_main(imgs_data, net)
    return result
#    return imgs_data, result

def load_imgs_ct(imgs_path):
    slices = []
    for s in os.listdir(imgs_path):
        try:
            one_slices = dicom.dcmread(os.path.join(imgs_path, s), force=True)
        except IOError:
            print('No such file')
            continue
        except InvalidDicomError:
            print('Invalid Dicom file')
            continue
        slices.append(one_slices)
    slices = [s for s in slices if s.Modality == 'CT']
    # 按照z轴坐标排序
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=False)
    sop_uid = [i.SOPInstanceUID for i in slices]
    Space = slices[0].PixelSpacing
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[0].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    temp_spacing = []
    for i in range(len(Space)):
        temp_spacing.append(float(Space[i]))
    spacing = map(float, ([slices[2].SliceThickness] + temp_spacing))
    spacing = np.array(list(spacing))

    #    spacing = slices[0].PixelSpacing
    #    spacing = np.array(spacing, dtype=np.float32)
    #        spacing顺序是xy
    #    spacing[0], spacing[1] = spacing[1], spacing[0]

    Space[0], Space[1] = Space[1], Space[0]

    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    image = image.astype(np.float32)
    image = image.swapaxes(0, 2)
    image[image < -1500] = -1024
    image[image > 2976] = 2976
    image = nearest_interpolation(image, Space, [1, 1])
    return image, sop_uid

# =============================================================================
# 导入网络
# =============================================================================
def load_net(weights_path):
    print ('start load model ......')
    with open(os.path.join(weights_path, 'full_body_classify.json')) as file:
        classify_net = models.model_from_json(file.read())
    classify_net.load_weights(os.path.join(weights_path, 'full_body_classify.h5'))
#   classify_net = load_model(os.path.join(weights_path, 'full_body_classify.h5'))
    print ('load model done ......')

    return classify_net