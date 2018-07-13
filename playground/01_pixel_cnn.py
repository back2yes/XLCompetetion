from torchvision.datasets import MNIST, EMNIST, FashionMNIST
import cv2

import os, time, sys
import numpy as np
import torch as tt
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
from glob import glob
from PIL import Image
from utils import dataloader
from torchvision.utils import make_grid, save_image

print("loading images...")
img_fp_list = sorted(glob("D:/XL/擦洞/*.jpg"))
xml_fp_list = sorted(glob("D:/XL/擦洞/*.xml"))
# img_fp_list.pop(29)
# xml_fp_list.pop(29)
imgs = [transforms.ToTensor()(Image.open(img_fp)) for img_fp in img_fp_list]
imgs = tt.stack(imgs)
xmls = [dataloader.parse_xml(xml_fp) for xml_fp in xml_fp_list]
print("finished.")


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {"A", "B"}
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        # self.mask[:, :, kH // 2, kW // 2 + (mask_type == "B"):] = 0
        # self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


fm = 64
net = nn.Sequential(
    MaskedConv2d("A", 1, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    # MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    nn.Conv2d(fm, 256, 1))
# print(net)
net.cuda()

print("building dataset...")
ds = dataloader.XLDataset(imgs, xmls, psize=112)
dl = DataLoader(ds, batch_size=1)
print("finished.")

# for xs, ts in dl:
#     img = make_grid(xs)
#     print(img.size())
#     cv2.imshow("hehe", img.numpy().transpose([1, 2, 0])[..., ::-1])
#     cv2.waitKey(100)
crit = nn.CrossEntropyLoss(reduce=False)
opti = optim.Adam(net.parameters(), lr=1e-4)

for epoch in range(10):
    for ii, (xs, ts) in enumerate(dl):
        print(ii)
        xs, ts = xs.mean(1, keepdim=True).cuda(), ts.cuda()
        ys = net(xs)
        # print(ts.size())
        # print(xs.size())
        loss = crit(input=ys, target=(xs[:, 0] * 255.0).long()) * (1.0 - ts[:, None, None, None].float() * 2.0)
        # loss = crit(input=ys, target=(xs[:, 0] * 255.0).long())
        net.zero_grad()
        # loss.abs().mean().backward()
        loss.mean().backward()
        opti.step()
        if ii % 10 == 0:
            print(loss.mean(1).mean(1).mean(1))
        if ii % 10 == 0:
            sample_ys = ys.cpu()
            sample_ys = tt.argmax(sample_ys, dim=1, keepdim=True) / 255.0
            # print(ys[0, 0])
            grid = make_grid(sample_ys)
            cv2.imshow("hehe", grid.numpy().transpose([1, 2, 0])[..., ::-1])
            cv2.waitKey(100)