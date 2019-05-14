import os
import math
import random
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import dataset
import dssnet

from albumentations import (ToFloat,
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion,
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness,
    Flip, OneOf, Compose, Resize)
from albumentations.pytorch import ToTensor

# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

IS_CUDA = True

def fit(epoch, model, data_loader, phase='training', optimizer=None):
    if phase == 'training':
        model.train()
    elif phase == 'validation':
        model.eval()
    running_loss = 0.0
    running_correct = 0
    # print("Len dataset", len(data_loader.dataset))
    # print("Model", model)
    # for param in model.classifier.parameters():
    #     print(param.requires_grad)
    # raise SystemExit

    for batch_idx, (data, target) in enumerate(data_loader):
        if IS_CUDA:
            data, target = data.cuda(), target.cuda()
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        # print(model.classifier)
        # print(output.shape)
        loss = criterion(output, target)
        if math.isnan(loss.item()) or math.isinf(loss.item()):
            raise
        #print("Loss", loss)
        running_loss += loss.item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        current_correct = preds.eq(target.data.view_as(preds)).cpu().sum()
        # print("Current correct", current_correct)
        running_correct += current_correct
        # print(running_correct)
        # print(target.data.view_as(preds))
        if phase == 'training':
            loss.backward()
            optimizer.step()
    print(running_loss)
    loss = running_loss/len(data_loader.dataset)
    accuracy = torch.tensor(100.) * running_correct/len(data_loader.dataset)
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is ' \
          f'{running_correct}/{len(data_loader.dataset)}={accuracy.item():{10}.{4}}')
    return loss, accuracy

def main():
    vgg = torchvision.models.vgg16(pretrained=True)

    for param in vgg.features.parameters():
        param.requires_grad = False

    vgg.classifier[6] = torch.nn.Linear(4096, 2)
    model = dssnet.build_model()

    if IS_CUDA:
        model = model.to('cuda')
        model = torch.nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []

    for epoch in range(1, 10):
        epoch_loss, epoch_accuracy = fit(epoch, model, train_loader, phase='training', optimizer=optimizer)
        val_epoch_loss, val_epoch_accuracy = fit(epoch, model, test_loader, phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        plt.plot(range(len(train_losses)), train_losses, 'bo')
        plt.plot(range(len(val_losses)), val_losses, 'r')

if __name__ == "__main__":
    main()
