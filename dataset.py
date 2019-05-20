from pathlib import Path
from typing import List

import random
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

MODEL_INPUT_SIZE = 224


class ImageData(data.Dataset):
    def __init__(self, files: List[Path], transform, t_transform):
        self.files = files
        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        file_path = self.files[item]
        image = Image.open(file_path)
        dir_path = file_path.parent
        wo_ext = file_path.stem
        label_path = (dir_path / wo_ext)
        label_path = label_path.with_suffix(label_path.suffix + '.png')
        label = Image.open(label_path).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        return image, label

    def __len__(self):
        return len(self.files)


def transform_to_input():
    return transforms.Compose([
        transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_loaders_hk(img_root, batch_size, num_thread=4, pin=True):
    transform = transform_to_input()
    t_transform = transforms.Compose([
        transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.round(x))
    ])
    random.seed(1001)
    img_root = Path(img_root)
    print(f"Image root dir {img_root}")
    files = list(img_root.glob('*.jpg'))
    print(f"Full size {len(files)}")
    train_size = int(0.8 * len(files))
    # test_size = len(files) - train_size
    print(f"Train size {train_size}")
    random.shuffle(files)
    train_dataset = ImageData(files[:train_size], transform, t_transform)
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_thread,
        pin_memory=pin)
    val_dataset = ImageData(files[train_size:], transform, t_transform)
    val_data_loader = data.DataLoader(
        dataset=val_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_thread,
        pin_memory=pin)
    return train_data_loader, val_data_loader


if __name__ == '__main__':
    # import numpy as np
    # img_root = '/home/ace/data/MSRA-B/image'
    # label_root = '/home/ace/data/MSRA-B/annotation'
    # filename = '/home/ace/data/MSRA-B/train_cvpr2013.txt'
    # loader = get_loader(img_root, label_root, 224, 1, filename=filename, mode='test')
    # for image, label in loader:
    #     print(np.array(image).shape)
    #     break
    img_root = Path('/home/hovnatan/work/MSRA-B')
    loader, _ = get_loaders_hk(img_root, 224, 1)
    for i, (image, label) in enumerate(loader):
        print(image.shape, image.dtype, torch.max(image[:, 1, :, :]))
        print(label.shape, label.dtype, torch.max(label))
        if i >= 10:
            break
