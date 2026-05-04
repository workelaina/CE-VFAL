from collections import OrderedDict
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
import torchvision
from torchvision.models import ResNet


class Client_L2(Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        embedding_size: int = 128
    ) -> None:
        super().__init__()
        activ = nn.ReLU(True)
        self.conv1 = nn.Linear(input_size, embedding_size)
        self.layer_one = nn.Sequential(OrderedDict([
            ('conv1', self.conv1),
            ('relu1', activ)
        ]))

        self.adv_inp = None
        self.layer_one_out = None

        self.other_layers = nn.ModuleList()
        self.fc1 = nn.Linear(embedding_size, output_size)
        self.layer_p = nn.Sequential(OrderedDict([
            ('fc1', self.fc1)
        ]))
        self.other_layers.append(self.layer_p)

    def fwd1(self, x: Tensor) -> Tensor:
        # self.adv_inp = x
        # self.adv_inp.requires_grad_()
        # self.adv_inp.retain_grad()
        x = self.layer_one(x)
        self.layer_one_out = x
        self.layer_one_out.requires_grad_()
        self.layer_one_out.retain_grad()
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.fwd1(x)
        x = self.layer_p(x)
        return x


class Client_L1(Client_L2):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        embedding_size: int = 128
    ):
        super().__init__(input_size, output_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fwd1(x)
        return x


class Server_L2(Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        embedding_size: int = 128
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        # x = torch.sigmoid(x)
        return x


def myResNet(
    resnet_num: Literal[18, 34, 50, 101, 152],
    in_channels: int,
    output_size: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> ResNet:
    # resnet: 3 7 2 3
    # hashvfl: 1 3 1 1

    if resnet_num == 18:
        f = torchvision.models.resnet18
    elif resnet_num == 34:
        f = torchvision.models.resnet34
    elif resnet_num == 50:
        f = torchvision.models.resnet50
    elif resnet_num == 101:
        f = torchvision.models.resnet101
    elif resnet_num == 152:
        f = torchvision.models.resnet152
    else:
        raise ValueError('What is resnet%d' % resnet_num)

    model = f(
        num_classes=output_size,
        weights=False
    )

    model.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=model.conv1.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )
    model.layer_one = model.conv1

    model.other_layers = nn.ModuleList()
    model.other_layers.append(model.bn1)
    model.other_layers.append(model.relu)
    # model.other_layers.append(model.maxpool)
    model.other_layers.append(model.layer1)
    model.other_layers.append(model.layer2)
    model.other_layers.append(model.layer3)
    model.other_layers.append(model.layer4)
    # model.other_layers.append(model.avgpool)
    model.other_layers.append(model.fc)

    def _forward_impl(self: ResNet, x: Tensor) -> Tensor:
        # self.adv_inp = x
        # self.adv_inp.requires_grad_()
        # self.adv_inp.retain_grad()
        x = self.layer_one(x)
        self.layer_one_out = x
        self.layer_one_out.requires_grad_()
        self.layer_one_out.retain_grad()

        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        x = F.avg_pool2d(x, 4, padding=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    from types import MethodType
    model._forward_impl = MethodType(_forward_impl, model)
    return model
