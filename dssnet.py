import sys
import traceback
from collections import OrderedDict

import torch
from torch import nn
import torchvision

from torch.utils.tensorboard import SummaryWriter

def pdb_excepthook(exc_type, exc_val, exc_tb):
    traceback.print_exception(exc_type, exc_val, exc_tb)
    __import__("ipdb").post_mortem(exc_tb)

sys.excepthook = pdb_excepthook

writer = SummaryWriter()

base = {'dss': [64, 64,
                'M', 128, 128,
                'M', 256, 256, 256,
                'M', 512, 512, 512,
                'M', 512, 512, 512,
                'M'
                ]}
extra = {'dss': [(64, 128, 3, [8, 16, 32, 64]),
                 (128, 128, 3, [4, 8, 16, 32]),
                 (256, 256, 5, [8, 16]),
                 (512, 256, 5, [4, 8]),
                 (512, 512, 5, []),
                 (512, 512, 7, [])
                 ]}
connect = {'dss': [[2, 3, 4, 5], [2, 3, 4, 5],
                   [4, 5],
                   [4, 5],
                   [],
                   []
                   ]}

# vgg16
def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    result = []
    for ind, layer in enumerate(layers):
        result.append((str(ind), layer))
    result = nn.Sequential(OrderedDict(result))
    return result

def vgg_def():
    vgg = torchvision.models.vgg16(pretrained=True)
    return vgg


# feature map before sigmoid: build the connection and deconvolution
class ConcatLayer(nn.Module):
    def __init__(self, list_k, k, scale=True):
        super(ConcatLayer, self).__init__()
        l, up, self.scale = len(list_k), [], scale
        for i in range(l):
            up.append(nn.ConvTranspose2d(1, 1, list_k[i], list_k[i] // 2, list_k[i] // 4))
        self.upconv = nn.ModuleList(up)
        self.conv = nn.Conv2d(l + 1, 1, 1, 1)
        self.deconv = nn.ConvTranspose2d(1, 1, k * 2, k, k // 2) if scale else None

    def forward(self, x, list_x):
        elem_x = [x]
        for i, elem in enumerate(list_x):
            elem_x.append(self.upconv[i](elem))
        if self.scale:
            out = self.deconv(self.conv(torch.cat(elem_x, dim=1)))
        else:
            out = self.conv(torch.cat(elem_x, dim=1))
        return out


# extend vgg: side outputs
class FeatLayer(nn.Module):
    def __init__(self, in_channel, channel, k):
        super(FeatLayer, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channel, channel, k, 1, k // 2),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, channel, k, 1, k // 2),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, 1, 1, 1))

    def forward(self, x):
        return self.main(x)


# fusion features
class FusionLayer(nn.Module):
    def __init__(self, nums=6):
        super(FusionLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(nums))
        self.nums = nums
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        for i in range(self.nums):
            out = self.weights[i] * x[i] if i == 0 else out + self.weights[i] * x[i]
        return out


# extra part
def extra_layer(vgg, cfg):
    feat_layers, concat_layers, scale = [], [], 1
    for k, v in enumerate(cfg):
        # side output (paper: figure 3)
        feat_layers += [FeatLayer(v[0], v[1], v[2])]
        # feature map before sigmoid
        concat_layers += [ConcatLayer(v[3], scale, k != 0)]
        scale *= 2
    return vgg, feat_layers, concat_layers


# DSS network
# Note: if you use other backbone network, please change extract
class DSS(nn.Module):
    def __init__(self, base, feat_layers, concat_layers, connect,
                 extract=(3, 8, 15, 22, 29), v2=True):
        super(DSS, self).__init__()
        self.extract = extract
        self.connect = connect
        basel = []
        for l in base:
            basel.append(l)
        self.base = nn.ModuleList(basel)
        for ind, m in enumerate(self.base):
            setattr(self, f"vgg{ind}", m)
        self.feat = nn.ModuleList(feat_layers)
        for ind, m in enumerate(self.feat):
            setattr(self, f"feat{ind}", m)
        self.comb = nn.ModuleList(concat_layers)
        for ind, m in enumerate(self.comb):
            setattr(self, f"concat{ind}", m)
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.v2 = v2
        if v2:
            self.fuse = FusionLayer()

    def forward(self, x):
        back, y, num = list(), list(), 0
        for k, _ in enumerate(self.base):
            x = self.base[k](x)
            if k in self.extract:
                y.append(self.feat[num](x))
                num += 1
        # side output
        y.append(self.feat[num](self.pool(x)))
        for i, _ in enumerate(y):
            back.append(self.comb[i](y[i], [y[j] for j in self.connect[i]]))
        # fusion map
        if self.v2:
            # version2: learning fusion
            back.append(self.fuse(back))
        else:
            # version1: mean fusion
            back.append(torch.cat(back, dim=1).mean(dim=1, keepdim=True))
        # add sigmoid
        return [torch.sigmoid(i) for i in back]


# build the whole network
def build_model():
    return DSS(*extra_layer(vgg_def().features, extra['dss']), connect['dss'])

def build_model_old():
    return DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']), connect['dss'])


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # vgg1 = vgg(base['dss'], 3)
    # print(vgg1)
    # vgg2 = vgg_def()
    # print(vgg2)
    # raise
    net = build_model()
    img = torch.randn(2, 3, 64, 64)
    out = net(img)
    for o in out:
        print(o.shape)
    # for param in net.parameters():
    #     print(param)
    writer.add_graph(net, img, verbose=False)
    writer.close()
