"""CIFAR ResNet-18 with optional 2D LOWO downsampling (B0 / A2 / A5 / A6).

Haar sites are only the three stride-2 stages (layer2/3/4): each has a
main-path HaarDownsample and a skip HaarDownsample (6 sites).  conv1 is
3×3 stride-1, so there is no stem DWT.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from lowo2d import HaarDownsample2D


def conv3x3(inp: int, out: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        haar: dict | None = None,
    ):
        super().__init__()
        self.use_haar = haar is not None and stride == 2
        if self.use_haar:
            self.conv1 = HaarDownsample2D(in_planes, planes, **haar)
            self.downsample = nn.Sequential(
                HaarDownsample2D(in_planes, planes, **haar),
                nn.BatchNorm2d(planes),
            )
        else:
            self.conv1 = conv3x3(in_planes, planes, stride)
            if stride != 1 or in_planes != planes:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
            else:
                self.downsample = None
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = x if self.downsample is None else self.downsample(x)
        return F.relu(out + residual)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, haar: dict | None = None):
        super().__init__()
        self.in_planes = 64
        self.haar = haar
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock(self.in_planes, planes, stride, self.haar)]
        self.in_planes = planes
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock(self.in_planes, planes, 1, None))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        return self.fc(x.flatten(1))


def build_model(arm: str, num_classes: int = 10) -> nn.Module:
    arm = arm.upper()
    if arm == "B0":
        return ResNet18CIFAR(num_classes=num_classes, haar=None)
    if arm == "A2":
        haar = dict(learnable=False, mode="so4", grouped_mix=True)
        return ResNet18CIFAR(num_classes=num_classes, haar=haar)
    if arm == "A5":
        haar = dict(learnable=True, mode="so4", grouped_mix=False)
        return ResNet18CIFAR(num_classes=num_classes, haar=haar)
    if arm == "A6":
        haar = dict(learnable=True, mode="so4", grouped_mix=True)
        return ResNet18CIFAR(num_classes=num_classes, haar=haar)
    if arm == "HWD":
        haar = dict(learnable=False, mode="so4", grouped_mix=False)
        return ResNet18CIFAR(num_classes=num_classes, haar=haar)
    if arm == "HWDQ":
        haar = dict(learnable=True, mode="so4", grouped_mix=False)
        return ResNet18CIFAR(num_classes=num_classes, haar=haar)
    raise ValueError(f"unknown arm {arm!r}; use B0, A2, A5, A6, HWD, or HWDQ")
