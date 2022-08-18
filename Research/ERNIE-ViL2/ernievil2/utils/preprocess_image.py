import os
import cv2
import sys
import numpy as np
import random
import time
import math
import json
import logging
import functools
import base64
from paddle.vision.transforms import ColorJitter,Grayscale

from paddle.vision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

def image_resize(img, width, height, interpolation=cv2.INTER_LINEAR):
    """
    resize image

    """
    img_resized = cv2.resize(img, (width, height), interpolation=interpolation)
    return img_resized



def group_resize(imgs, w, h):
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        img = image_resize(img, w, h)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        resized_imgs.append(img)
    return resized_imgs

def group_scale(imgs, target_size):
    """
    scale batch images
    """
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        h, w, c = img.shape
        if (w <= h and w == target_size) or (h <= w and h == target_size):
            resized_imgs.append(img)
            continue

        if w < h:
            ow = target_size
            # oh = int(target_size * 4.0 / 3.0)
            oh = int(float(target_size) / w * h) + 1
            resized_imgs.append(image_resize(img, ow, oh))
        else:
            oh = target_size
            # ow = int(target_size * 4.0 / 3.0)
            ow = int(float(target_size) / h * w) + 1
            resized_imgs.append(image_resize(img, ow, oh))
    return resized_imgs

def group_random_crop(img_group, target_size):
    """
    crop batch images randomly

    """
    h, w, c = img_group[0].shape
    th, tw = target_size, target_size

    out_images = []
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for img in img_group:
        if w == tw and h == th:
            out_images.append(img)
        else:
            out_images.append(img[y1:(y1 + th), x1:(x1 + tw)])

    return out_images
def group_random_flip(img_group):
    """
    flip batch images randomly

    """
    v = random.random()
    if v < 0.5:
        ret = [cv2.flip(img, 1) for img in img_group]
        return ret
    else:
        return img_group


def group_center_crop(img_group, target_size):
    """
    crop batch images' center
    """
    img_crop = []
    for img in img_group:
        h, w, c = img.shape
        th, tw = target_size, target_size
        x1 = math.ceil((w - tw) / 2.)
        y1 = math.ceil((h - th) / 2.)
        img_crop.append(img[y1:(y1 + th), x1:(x1 + tw)])

    return img_crop


def imageloader(buf):
    """
    load image

    """
    arr = np.fromstring(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


def new_data_augument(imgs):
    new_imgs = []
    jitter_transform = ColorJitter(0.4, 0.4, 0.4, 0.1)
    grayscale_transform = Grayscale(num_output_channels=3)
    for img in imgs:
        p_jitter = random.random()
        p_grayscale = random.random()
        p_blur = random.random()
        if p_jitter < 0.8:
            img  = jitter_transform(img)
        if p_grayscale < 0.2:
            img = grayscale_transform(img)
        if p_blur < 0.5:
            sigma = random.uniform(0.1,2.0)
            img = cv2.GaussianBlur(img, (5,5), sigma)
        new_imgs.append(img)
    return new_imgs
def decode_image_base64(img,
                 mode="train",
                 seg_num=1,
                 short_size=224,
                 crop_size=224,
                 data_augument=False):
    """
    decode video and extract features

    """
    try: 
        imgstr = base64.b64decode(img)
        imgs = [imageloader(imgstr)]
        if data_augument:
            imgs = group_scale(imgs, short_size)
            if mode == 'train':
                imgs = group_random_crop(imgs, crop_size)
                imgs = group_random_flip(imgs)
                imgs = new_data_augument(imgs)
            else:
                imgs = group_center_crop(imgs, crop_size)
        else:
            imgs = group_resize(imgs, crop_size, crop_size)
            # if mode == 'train':
            #     imgs = group_random_flip(imgs)
        np_imgs = (np.array(imgs[0]).astype('float32').transpose(
            (2, 0, 1))).reshape(1, 3, crop_size, crop_size) / 255
        for i in range(len(imgs) - 1):
            img = (np.array(imgs[i + 1]).astype('float32').transpose(
                (2, 0, 1))).reshape(1, 3, crop_size, crop_size) / 255
            np_imgs = np.concatenate((np_imgs, img))
        imgs = np_imgs
        imgs -= img_mean
        imgs /= img_std
        return 0, imgs
    except Exception as e:
        print(">>> hello_debug: decode base64 images error {}".format(e))
        return 1, None






