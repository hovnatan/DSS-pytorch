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

class DogsAndCatsTrainingDataset(Dataset):
    def __init__(self, files, aug=False):
        self.files = files
        self.class_to_idx = {'cat' : 0, 'dog' : 1}
        self.classes = ['cat', 'dog']
        self.to_torch = Compose([Resize(224, 224),
                                 ToTensor({"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})])
        self.aug = aug
        self.aug_tr = OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            RandomRotate90(),
            IAAEmboss(),
            Transpose(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        filepath = self.files[idx]
        label = os.path.basename(filepath).split('.')[-3]
        img = Image.open(filepath)
        img = np.array(img)
        if self.aug:
            img = self.aug_tr(image=img)['image']
        img = self.to_torch(image=img)['image']

        return img, self.class_to_idx[label]


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

    criterion = torch.nn.CrossEntropyLoss()
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
    random.seed(1001)
    root_dir = '/home/hovnatan/work/train'
    files = glob(os.path.join(root_dir, '*.jpg'))
    train_size = int(0.8 * len(files))
    # test_size = len(files) - train_size
    random.shuffle(files)

    train_dataset = DogsAndCatsTrainingDataset(files[:train_size],
                                               True)
    test_dataset = DogsAndCatsTrainingDataset(files[train_size:],
                                              False)

    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=8)

    vgg = torchvision.models.vgg16(pretrained=True)

    for param in vgg.features.parameters():
        param.requires_grad = False

    vgg.classifier[6] = torch.nn.Linear(4096, 2)
    model = vgg

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
