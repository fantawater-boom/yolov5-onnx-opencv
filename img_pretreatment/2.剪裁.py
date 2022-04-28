# coding: utf-8
import copy

from PIL import Image
import os
import os.path
import numpy as np
import cv2
"""
本文件完成：剪裁图像，将1080P的图像按照固定的比例剪裁成6个416X416的图像
"""
# 指明被遍历的文件夹
rootdir = r'E:\yiyasuo'
for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
    for filename in filenames:
        print('parent is :' + parent)
        print('filename is :' + filename)
        currentPath = os.path.join(parent, filename)
        print('the fulll name of the file is :' + currentPath)

        img = Image.open(currentPath)
        print(img.format, img.size, img.mode)


        # 开始剪裁
        list_size = [{"a": (124, 128), "b": (540, 544)},
                     {"a": (540, 128), "b": (956, 544)},
                     {"a": (124, 544), "b": (540, 960)},
                     {"a": (540, 544), "b": (956, 960)},
                     {"a": (124, 960), "b": (540, 1376)},
                     {"a": (540, 960), "b": (956, 1376)},
                     {"a": (124, 1376), "b": (540, 1792)},
                     {"a": (540, 1376), "b": (956, 1792)}]

        for i in range(8):
            tempimg = copy.deepcopy(img)
            t1 = list_size[i].get("a")[0]
            t2 = list_size[i].get("a")[1]
            t3 = list_size[i].get("b")[0]
            t4 = list_size[i].get("b")[1]
            box = (t1, t2, t3, t4)
            image1 = tempimg.crop(box)
            split_list = filename.split(".")# 图像裁剪
            image1.save(r"E:\2" + '\\' + split_list[0]+"_"+str(i)+"."+split_list[1])  # 存储裁剪得到的图像
        # box1 = (400, 400, 800, 800)  # 设置左、上、右、下的像素
