# coding=utf-8

import torch.nn as nn
import torch.nn.functional as F

# class simpleconv3(nn.Module):
#     ## 初始化函数
#     def __init__(self,nclass):
#         super(simpleconv3,self).__init__()
#         self.conv1 = nn.Conv2d(3, 12, 3, 2) #输入图片大小为3*48*48，输出特征图大小为12*23*23，卷积核大小为3*3，步长为2
#         self.bn1 = nn.BatchNorm2d(12)
#         self.conv2 = nn.Conv2d(12, 24, 3, 2) #输入图片大小为12*23*23，输出特征图大小为24*11*11，卷积核大小为3*3，步长为2
#         self.bn2 = nn.BatchNorm2d(24)
#         self.conv3 = nn.Conv2d(24, 48, 3, 2) #输入图片大小为24*11*11，输出特征图大小为48*5*5，卷积核大小为3*3，步长为2
#         self.bn3 = nn.BatchNorm2d(48)
#         self.fc1 = nn.Linear(48 * 5 * 5 , 1200) #输入向量长为48*5*5=1200，输出向量长为1200
#         self.fc2 = nn.Linear(1200, 128) #输入向量长为1200，输出向量长为128
#         self.fc3 = nn.Linear(128, nclass) #输入向量长为128，输出向量长为nclass，等于类别数
#
#     ## 前向函数
#     def forward(self, x):
#         ## relu函数，不需要进行实例化，直接进行调用
#         ## conv，fc层需要调用nn.Module进行实例化
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = x.view(-1, 48 * 5 * 5)  # reshape
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)

class FaceCNN(nn.Module):
    # 初始化网络结构
    def __init__(self, nclass):
        super(FaceCNN, self).__init__()

        # 第一层卷积、池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.BatchNorm2d(num_features=64),  # 归一化
            nn.RReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大值池化
        )

        # 第二层卷积、池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            # output:(bitch_size, 128, 12 ,12)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第三层卷积、池化
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            # output:(bitch_size, 256, 6 ,6)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 参数初始化
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=128, out_features=nclass),
        )

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 数据扁平化
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y



if __name__ == '__main__':
    import torch
    x = torch.randn(1,3,48,48)
    model = FaceCNN(6)
    y = model(x)
    print(model)