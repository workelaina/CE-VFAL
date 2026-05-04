import sys
from typing import Iterable

import torch
from torch import nn, Tensor
from torch.nn import Module


class Hamiltonian(nn.modules.loss._Loss):
    def __init__(self, layer: Module, reg_cof: float = 1e-4) -> None:
        super(Hamiltonian, self).__init__()
        self.layer = layer
        self.reg_cof = reg_cof

    def forward(self, x: Tensor, p: Tensor) -> Tensor:
        y = self.layer(x)
        h = torch.sum(y * p)
        return h


class CrossEntropyWithWeightPenlty(nn.modules.loss._Loss):
    def __init__(
        self,
        device: str,
        module: Module,
        reg_cof: float = 1e-4
    ) -> None:
        super(CrossEntropyWithWeightPenlty, self).__init__()
        self.module = module
        self.reg_cof = reg_cof
        self.criterion = nn.CrossEntropyLoss().to(device)

    def __call__(self, pred: Tensor, label: Tensor) -> Tensor:
        cross_loss = self.criterion(pred, label)
        weight_loss = cal_l2_norm(self.module)
        loss = cross_loss + self.reg_cof * weight_loss
        return loss


def cal_l2_norm(layer: Module) -> Tensor:
    loss = 0.
    for name, param in layer.named_parameters():
        if name == 'weight':
            loss += 0.5 * torch.norm(param) ** 2
    return loss


def gen_eta(
    device: str,
    shape: Iterable[int],
    eps: float,
    random_start: bool = True
) -> Tensor:
    if random_start:
        eta = torch.FloatTensor(
            *shape
        ).uniform_(
            -eps,
            eps
        ).to(device)
    else:
        eta = torch.zeros(
            *shape
        ).to(device)

    return eta
