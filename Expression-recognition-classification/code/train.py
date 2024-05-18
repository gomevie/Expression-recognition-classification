# coding=utf-8

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os
import numpy as np
from net import FaceCNN

## 使用tensorboardX进行可视化
from tensorboardX import SummaryWriter
writer = SummaryWriter('../code') ## 创建一个SummaryWriter的示例，默认目录名字为runs

## 训练主函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  #训练模式
            else :
                model.train(False)  #验证模式

            running_loss = 0.0
            running_accs = 0.0
            number_batch = 0

            ## 从dataloaders中获得数据
            for data in dataloaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad() ##清空梯度
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data.item()
                running_accs += torch.sum(preds == labels).item()
                number_batch += 1

            epoch_loss = running_loss / number_batch
            epoch_acc = running_accs  / dataset_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/trainloss', epoch_loss, epoch)
                writer.add_scalar('data/trainacc', epoch_acc, epoch)
            else:
                writer.add_scalar('data/valloss', epoch_loss, epoch)
                writer.add_scalar('data/valacc', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            scheduler.step()

    writer.close()
    return model


if __name__ == '__main__':
    ## 参数设定
    image_size = 48
    crop_size = 48
    nclass = 7
    batch_size = 256
    model = FaceCNN(nclass)

    # 加载模型
    try:
        model.load_state_dict(torch.load('../models/model.pt'))
        model.eval()
    except:
        pass

    data_dir = '../data'  ## 数据目录

    ## 模型缓存接口
    if not os.path.exists('../models'):
        os.mkdir('../models')

    ## 检查gpu是否可用
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    print(model)

    ## 创建数据预处理函数，训练预处理包括随机裁剪缩放、随机翻转、归一化，验证预处理包括中心裁剪，归一化
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomCrop(48),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(48),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(64),
            # transforms.CenterCrop(48),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    }

    ## 使用torchvision的dataset ImageFolder接口读取数据
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}



    ## 创建数据指针，设置batch大小，shuffle，多进程数量
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4) for x in ['train', 'val']}
    ## 获得数据集大小
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    ## 优化目标使用交叉熵，优化方法使用带动量项的SGD，学习率迭代策略为step，每隔100个epoch，变为原来的0.1倍
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

    model = train_model(model=model,
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epochs=30)

    torch.save(model.state_dict(), '../models/model.pt')

