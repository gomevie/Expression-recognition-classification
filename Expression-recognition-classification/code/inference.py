# coding=utf-8

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
import cv2

test_size = 48
from net import FaceCNN
net = FaceCNN(7)
net.eval()
torch.no_grad()

modelpath = '../models/model.pt'
net.load_state_dict(torch.load(modelpath,map_location=lambda storage,loc: storage))

data_transforms =  transforms.Compose([
            # transforms.Resize(64),
            # transforms.CenterCrop(48),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])



def geturlPath():
    # 指定路径
    path = r'../data/val/angry/'
    # 返回指定路径的文件夹名称
    dirs = os.listdir(path)
    # 循环遍历该目录下的照片
    for dir in dirs:
        # 拼接字符串
        pa = path+dir
        # 判断是否为照片
        if not os.path.isdir(pa):
            # 使用生成器循环输出
            yield pa

nums = 0
accs = 0
for imagepath in geturlPath():
    # print(imagepath)

    nums += 1
    image = Image.open(imagepath)
    imgblob = data_transforms(image).unsqueeze(0)

    predict = net(imgblob)
    index = np.argmax(predict.detach().numpy())
    # print(predict)
    # print(index)
    if index == 0:
       accs += 1

print(f"acc: {accs * 100 / nums} %")

