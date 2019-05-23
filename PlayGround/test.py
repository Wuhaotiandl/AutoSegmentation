from core.shelper import *

def test(img_path, seg_path):
    data = ResampleDcm(img_path, outputspacing =[1.1, 1.1, 1])
    masks = ResampleDcm(seg_path, outputspacing=[1.1, 1.1, 1])
    for i in range(data.shape[0]):
        image = data[i]
        mask = masks[i]
        image[image < -150] = -150
        image[image > 200] = 200
        mask[mask >= 1] = 200
        ShowImage(1, image, mask)

img_path = r'H:\LiverData\LITS17\volumes\volume-0.nii'
seg_path = r'H:\LiverData\LITS17\segmentations\segmentation-0.nii'
test(img_path, seg_path)