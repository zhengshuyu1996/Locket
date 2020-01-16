import os
from PIL import Image, ImageEnhance
import numpy as np 
import sys 
import skimage 

def preprocess_for_sketch(img):
    # 转为底色白色，黑白照片
    # 输入图片是rgba格式 
    im = np.array(img)
    alpha = im[:, :, 3]
    mask = np.expand_dims(np.where(alpha > 0, 1, 0), axis=2)
    mask = np.tile(mask, (1, 1, 4))
    im = mask * im
    tmp = (1 - mask) * 255
    im = im + tmp
    im = im[:, :, 0:3].astype(np.uint8)
    img = Image.fromarray(im)
    img = img.convert('L')
    return img
    
def preprocess_for_painting(img, brightness=1.1, contrast=1.3):
    im = np.array(img)
    alpha = im[:, :, 3]
    mask = np.expand_dims(np.where(alpha > 0, 1, 0), axis=2)
    mask = np.tile(mask, (1, 1, 4))
    im = mask * im
    im = im[:, :, 0:3].astype(np.uint8)
    img = Image.fromarray(im)
    
    #亮度增强
    enh_bri = ImageEnhance.Brightness(img)
    img = enh_bri.enhance(brightness)

    #对比度增强
    enh_con = ImageEnhance.Contrast(img)
    img = ImageEnhance.Contrast(img)
    img = enh_con.enhance(contrast)

    return img

def postprocess_for_painting(img):
    # 是否需要弱化一下边缘？
    return img
