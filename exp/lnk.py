from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Module

from com import tensor_size
from network import Client_L1, Server_L2
from utils import get_inputs_list

autoatk_communicate = 0


class fo_communicate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor):
        global autoatk_communicate
        autoatk_communicate += tensor_size(x)
        return x.detach()

    @staticmethod
    def backward(ctx, x: Tensor):
        global autoatk_communicate
        autoatk_communicate += tensor_size(x)
        return x.detach()


class LnkAA(Module):
    def __init__(
        self,
        server: Server_L2,
        clients: Iterable[Client_L1],
        n_client: int,
        pad: bool
    ) -> None:
        super().__init__()
        self.server = server
        self.clients = clients
        self.n_client = n_client
        self.pad = pad

    def forward(self, inputs: Tensor) -> Tensor:
        inputs_list, _ = get_inputs_list(
            inputs,
            self.n_client,
            self.pad
        )

        outputs_list = [
            self.clients[i](j) for i, j in enumerate(inputs_list)
        ]
        outputs_client = torch.cat(outputs_list, dim=-1)
        # outputs_e, msg_size = self.compressor.c_dec(outputs_client)
        # self.last_res.communicate(msg_size)
        outputs_e = fo_communicate.apply(outputs_client)
        output = self.server(outputs_e)
        return output
