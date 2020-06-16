#!/user/bin/env python
# -*- coding:utf-8 -*-

import sys

sys.dont_write_bytecode = True

import os
import numpy as np

import cv2


# from pycocotools.coco import COCO
# from pycocotools import mask as maskUtils
from lib.config import Config
from lib.model import MaskRCNN
from lib import utils
from lib import visualize
from tools import getColor
from tools import ColorTest

from PIL import Image
from keras.backend import clear_session
# clear_session()

class DeepFashion2Config(Config):

    NAME = "deepfashion2"

    IMAGES_PER_GPU = 2

    GPU_COUNT = 1

    NUM_CLASSES = 1 + 13

    USE_MINI_MASK = True

def output(image,model):
    results = model.detect([image], verbose=1)
    r = results[0]

    clothclass = [] #存衣服的类别
    clothroi = [] #存衣服的边框
    clothimg = [] #存衣服的图片
    clothnum = 0 #存衣服的数量
    clothcol = [] #存衣服的颜色

    num1,num2,num3=0,0,0
    maxindex01 = -1 #上衣下标
    maxindex02 = -1 #下衣下标
    maxindex03 = -1 #下衣下标
    ###判断衣服数量
    for i in range(len(r['class_ids'])):
        #上衣
        if r['class_ids'][i] < 7:
            num1+=1
            maxindex01 = i
        #下衣
        if ((r['class_ids'][i]>6) and (r['class_ids'][i]<10)):
            num2+=1
            maxindex02 = i
        #裙子
        if r['class_ids'][i]>9:
            num3+=1
            maxindex03 = i
    #多件上衣的情况：
    if num1>1:
        maxscore = 0
        for i in range(len(r['class_ids'])):
            if r['class_ids'][i] < 7:
                if r['scores'][i]>maxscore:
                    maxscore = r['scores'][i]
                    maxindex01 = i
    # 多件下衣的情况：
    if num2>1:
        maxscore = 0
        for i in range(len(r['class_ids'])):
            if (r['class_ids'][i] < 10) and (r['class_ids'][i] > 6):
                if r['scores'][i]>maxscore:
                    maxscore = r['scores'][i]
                    maxindex02 = i
    # 多件裙子的情况：
    if num3>1:
        maxscore = 0
        for i in range(len(r['class_ids'])):
            if (r['class_ids'][i] > 9) :
                if r['scores'][i]>maxscore:
                    maxscore = r['scores'][i]
                    maxindex03 = i
    index = []

    if not maxindex01 == -1 :
        index.append(maxindex01)
        #clothclass.append(r['class_ids'][maxindex01])
        #clothroi.append(r['rois'][maxindex01])

    if not maxindex02 == -1 :
        index.append(maxindex02)
        #clothclass.append(r['class_ids'][maxindex02])
        #clothroi.append(r['rois'][maxindex02])

    if not maxindex03 == -1 :
        index.append(maxindex03)
        #clothclass.append(r['class_ids'][maxindex03])
        #clothroi.append(r['rois'][maxindex03])

    index.sort()

    for i in range(len(index)):
        clothclass.append(r['class_ids'][index[i]])
        clothroi.append(r['rois'][index[i]])

    clothnum = len(clothclass)

    N = r['rois'].shape[0]
    mask = []
    for i in range(N):
        mask.append(r['masks'][:, :, i])

    for i in range(len(clothroi)):
        w1,h1,w2,h2 = clothroi[i]
        img = image[w1:w2,h1:h2]
        clothimg.append(img)
        ###color
        masked = np.array(mask[i], dtype=np.uint8)
        masked = masked[w1:w2,h1:h2]
        masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=masked)
        clothcol.append(getColor.getColor(masked))
        #if i == 1:
            #ColorTest.getColor(masked)

    Clotheslist = []

    for i in range(len(clothclass)):
        item = {}
        item['classid']=int(clothclass[i])
        item['color'] = clothcol[i]
        item['colornum'] = len(clothcol[i])

        #jpg_as_bytes = base64.b64encode(clothimg[i])
        #jpg_as_str = jpg_as_bytes.decode('ascii')
        item['img'] = clothimg[i].tolist()  ##图片转为列表储存

        Clotheslist.append(item)

    backdata = {}
    backdata['num']=clothnum
    backdata['Clotheslist']=Clotheslist

    #print("clothcol:",clothcol)

    #print(backdata)

    return backdata

def getdata(img,MODEL):

    # clear_session()
    #
    # ROOT_DIR = os.path.abspath("./")
    # DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
    # WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_deepfashion2_0060.h5")
    #
    # # Configurations
    # class InferenceConfig(DeepFashion2Config):
    #     GPU_COUNT = 1
    #     IMAGES_PER_GPU = 1
    #
    # config = InferenceConfig()
    # # config.display()
    #
    # # Create model
    # model = MaskRCNN(mode="inference", config=config, model_dir="./logs/")
    #
    # # Select weights file to load 选择要加载的权重文件
    # weights_path = WEIGHTS_PATH
    # # weights_path = model.find_last()
    #
    # # Load weights 加载权重
    # print("Loading weights ", weights_path)
    # model.load_weights(weights_path, by_name=True)

    # Train or evaluate 训练或评估
    backdata = output(img,MODEL)

    return backdata

