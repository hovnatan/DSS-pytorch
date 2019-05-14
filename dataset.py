import os
import glob
import random
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

class ImageData_hk(data.Dataset):
    def __init__(self, files, transform, t_transform):
        self.files = files
        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        file_path = self.files[item]
        image = Image.open(file_path)
        dir_path = os.path.dirname(file_path)
        wo_ext = os.path.splitext(file_path)[0]
        label_path = os.path.join(dir_path, wo_ext + '.png')
        label = Image.open(label_path).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        return image, label

    def __len__(self):
        return len(self.files)

class ImageData(data.Dataset):
    """ image dataset
    img_root:    image root (root which contain images)
    label_root:  label root (root which contains labels)
    transform:   pre-process for image
    t_transform: pre-process for label
    filename:    MSRA-B use xxx.txt to recognize train-val-test data (only for MSRA-B)
    """

    def __init__(self, img_root, label_root, transform, t_transform, filename=None):
        if filename is None:
            self.image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
            self.label_path = list(
                map(lambda x: os.path.join(label_root, x.split('/')[-1][:-3] + 'png'), self.image_path))
        else:
            lines = [line.rstrip('\n')[:-3] for line in open(filename)]
            self.image_path = list(map(lambda x: os.path.join(img_root, x + 'jpg'), lines))
            self.label_path = list(map(lambda x: os.path.join(label_root, x + 'png'), lines))

        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        label = Image.open(self.label_path[item]).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        return image, label

    def __len__(self):
        return len(self.image_path)


# get the dataloader (Note: without data augmentation)
def get_loader(img_root, label_root, img_size, batch_size,
              filename=None, mode='train', num_thread=4, pin=True):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        dataset = ImageData(img_root, label_root, transform, t_transform, filename=filename)
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
                                      pin_memory=pin)
        return data_loader
    else:
        t_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        dataset = ImageData(img_root, label_root, None, t_transform, filename=filename)
        return dataset

def get_loader_hk(img_root, img_size, batch_size,
                 mode='train', num_thread=4, pin=True):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        random.seed(1001)
        files = glob.glob(os.path.join(img_root, '*.jpg'))
        train_size = int(0.8 * len(files))
        # test_size = len(files) - train_size
        random.shuffle(files)
        dataset = ImageData_hk(files[:train_size], transform, t_transform)
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
                                      pin_memory=pin)
        return data_loader


if __name__ == '__main__':
    # import numpy as np
    # img_root = '/home/ace/data/MSRA-B/image'
    # label_root = '/home/ace/data/MSRA-B/annotation'
    # filename = '/home/ace/data/MSRA-B/train_cvpr2013.txt'
    # loader = get_loader(img_root, label_root, 224, 1, filename=filename, mode='test')
    # for image, label in loader:
    #     print(np.array(image).shape)
    #     break
    img_root = '/home/hovnatan/work/MSRA-B'
    loader = get_loader_hk(img_root, 224, 1)
    for i, (image, label) in enumerate(loader):
        print(image.shape, image.dtype, torch.max(image[:, 1, :, :]))
        print(label.shape, label.dtype, torch.max(label))
        if i >= 10:
            break
