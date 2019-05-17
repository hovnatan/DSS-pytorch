import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision
import dataset
import dssnet
from loss import Loss
import albumentations


IS_CUDA = True


def fit(epoch, model, data_loader, phase='training', criterion=None,
        optimizer=None):
    if phase == 'training':
        model.train()
    elif phase == 'validation':
        model.eval()
    running_loss = 0.0
    running_mae = 0.0

    for (data, target) in data_loader:
        if IS_CUDA:
            data, target = data.cuda(), target.cuda()
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        # print(model.classifier)
        # print(output.shape)
        loss = criterion(output, target)
        running_loss += loss.item()
        prob_pred = torch.mean(torch.cat(output, dim=1), dim=1, keepdim=True)
        mae = torch.abs(prob_pred - target).mean()
        print(f'{phase}, epoch {epoch}, loss {loss.item()}, MAE {mae.item()}')
        running_mae += mae
        # preds = output.data.max(dim=1, keepdim=True)[1]
        # current_correct = preds.eq(target.data.view_as(preds)).cpu().sum()
        # print("Current correct", current_correct)
        # running_correct += current_correct
        # print(running_correct)
        # print(target.data.view_as(preds))
        if phase == 'training':
            loss.backward()
            optimizer.step()
    loss = running_loss/len(data_loader)
    mae = running_mae/len(data_loader)
    # accuracy = torch.tensor(100.) * running_correct/len(data_loader.dataset)
    # print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is ' \
          # f'{running_correct}/{len(data_loader.dataset)}={accuracy.item():{10}.{4}}')
    return loss, mae

def main(config):
    model = dssnet.build_model()

    if config.init_model:
        model.load_state_dict(torch.load(config.init_model))
    else:
        for module in model.modules():
            if module in model.base:
                # print(module)
                continue
            else:
                dssnet.weights_init(module)
    model.train()
    criterion = Loss()
    if IS_CUDA:
        model = model.to('cuda')
        criterion = criterion.to('cuda')
        # model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)
    img_root = Path.home() / 'work/MSRA-B'
    train_loader, test_loader = dataset.get_loaders_hk(img_root, 224, 12, 4,
                                                       pin=False)

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []

    for epoch in range(0, config.epochs_to_train):
        epoch_loss, epoch_accuracy = fit(epoch, model, train_loader, phase='training',
                                         criterion=criterion,
                                         optimizer=optimizer)
        with torch.no_grad():
            val_epoch_loss, val_epoch_accuracy = fit(
                epoch, model, test_loader, criterion=criterion, phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        print(
            f"Training accuracy {train_accuracy.item()}, val accuracy {val_accuracy.item()}")
    if config.save_model:
        torch.save(model.state_dict(), config.save_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_model', type=str)
    parser.add_argument('--save_model', type=str)
    parser.add_argument('--test_input_path', type=str)
    parser.add_argument('--test_output_path', type=str)
    parser.add_argument('--epochs_to_train', type=int, default=0)
    config = parser.parse_args()
    main(config)
