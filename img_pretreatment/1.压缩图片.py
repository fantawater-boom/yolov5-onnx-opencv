# coding: utf-8
from PIL import Image

import os.path

import cv2

"""
本文件完成图像的预处理：压缩图像到1080p

"""

# 指明被遍历的文件夹
rootdir = r'E:\1'
for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
    for filename in filenames:
        currentPath = os.path.join(parent, filename)
        img1 = cv2.imread(currentPath)
        img2 = cv2.resize(img1, (1080, 1920))
        cv2.imwrite(filename, img2)
        # box1 = (400, 400, 800, 800)  # 设置左、上、右、下的像素
