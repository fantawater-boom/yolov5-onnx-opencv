import os
import cv2
import glob
import random
#
train_txt_path = 'train.txt'
val_txt_path = 'val.txt'
#全部的txt
path_imgs = 'E:/test/*.jpg'
#glob.glob返回所有匹配的文件路径列表。
image_list = glob.glob(path_imgs)
#打乱
random.shuffle(image_list)
#这里是划分，我设置的是0.85：0.15  可以根据自己情况划分
num = len(image_list)
train_list = image_list[:int(0.85*num)]
val_list = image_list[int(0.85*num):]
#写入，CV2的判断语句是因为有些图片CV2无法读取，会返回none，导致报错，所以我们直接跳过这样的图片
with open(train_txt_path,'w') as f:
    for line in train_list:
        jpg_name = line
        # jpg_name = line.replace('txt','jpg')
        img = cv2.imread(jpg_name)
        if img is not None:
            f.write(jpg_name + '\n')
#写入验证集
with open(val_txt_path,'w') as f:
    for line in val_list:
        jpg_name = line.replace('txt','jpg')
        img = cv2.imread(jpg_name)
        if img is not None:
            f.write(jpg_name + '\n')
