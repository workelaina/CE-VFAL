import os
import sys
from typing import Iterable, Union, Callable
import numpy as np

import torch
from torch import Tensor

from cfgclass import Cfg
from com import Compressor
from lnk import LnkAA
from nvflnet import NVflNet
from party import Server, Client
from utils import get_inputs_list
from zofwd import Zoos

STR_CLIENT_MODEL = 'client%d.checkpoint'


class VflNet(NVflNet):
    def __init__(self, cfg: Cfg) -> None:
        super().__init__(cfg)
        self.zoos = Zoos(
            cfg.net.q,
            cfg.device,
            cfg.net.mu,
            cfg.net.u_type,
            [cfg.batch_size, cfg.net.msg_size * cfg.net.n_client]
        )
        self.compressor = Compressor(
            cfg.device,
            cfg.com.typ,
            cfg.com.bit
        )
        self.re_compressor = Compressor(
            cfg.device,
            cfg.com.re_typ,
            cfg.com.re_bit
        )

    def init_model(self, shape1: int) -> None:
        self.server = Server(self.cfg, self.cfg.net.output_size)
        self.model = self.server
        self.clients = [Client(
            self.cfg, shape1
        ) for _ in range(self.cfg.net.n_client)]
        self.lnkaa = LnkAA(
            self.server,
            self.clients,
            self.cfg.net.n_client,
            self.cfg.net.pad
        )

    def save_model(self, x: Union[str, int] = None) -> str:
        x = super().save_model(x)
        for i, client in enumerate(self.clients):
            client.save(os.path.join(x, STR_CLIENT_MODEL % i))
        return x

    def load_model(self, x: Union[str, int] = None) -> str:
        x = super().load_model(x)
        for i, client in enumerate(self.clients):
            client.load(os.path.join(x, STR_CLIENT_MODEL % i))
        return x

    def zero_grad(self) -> None:
        for m in range(self.cfg.net.n_client):
            self.clients[m].zero_grad()
        super().zero_grad()

    def step(self) -> None:
        for m in range(self.cfg.net.n_client):
            self.clients[m].step()
        super().step()

    def lr_step(self) -> None:
        for m in range(self.cfg.net.n_client):
            self.clients[m].lr_step()
        super().lr_step()

    def zero_grad_ce(self) -> None:
        for m in range(self.cfg.net.n_client):
            self.clients[m].zero_grad_ce()
        super().zero_grad()

    def step_ce(self) -> None:
        for m in range(self.cfg.net.n_client):
            self.clients[m].step_ce()
        super().step()

    def backward(
        self,
        inputs: Iterable[str] = None,
        retain_graph: bool = None
    ) -> None:
        self.server.backward(self.last_loss, None, None, retain_graph)
        for m in range(self.cfg.net.n_client):
            self.clients[m].backward(
                self.last_outputs_fwd,
                self.last_partial,
                inputs,
                retain_graph
            )

    def in2loss(
        self,
        inputs: Tensor = None,
        f: Callable[[Tensor], Tensor] = None,
        append_log: bool = False,
        only_loss: bool = False,
        zo: bool = True
    ) -> Tensor:
        if inputs is None:
            inputs = self.last_inputs
        inputs_list, _ = get_inputs_list(
            inputs,
            self.cfg.net.n_client,
            self.cfg.net.pad
        )
        labels = self.last_labels

        outputs_list = [
            self.clients[i](j) for i, j in enumerate(inputs_list)
        ]
        outputs_client = torch.cat(outputs_list, dim=-1)
        outputs_e, msg_size = self.compressor.c_dec(outputs_client)
        self.last_res.communicate(msg_size)

        self.last_outputs_fwd = outputs_client

        outputs_e.requires_grad_()
        output = self.server(outputs_e)

        if f is None:
            def f(_output: Tensor) -> Tensor:
                return self.loss_fn(_output, labels)

        loss = f(output)
        self.last_loss = loss
        self.last_partial = None

        if append_log:
            self.last_res.batch(output, self.last_labels, loss)
        if only_loss:
            self.last_loss = 'not_use'
            self.last_partial = 'not_use'
            return loss

        if zo:
            def g(_x: Tensor) -> Tensor:
                return f(self.server(_x))
            delta, us = self.zoos.zo_delta(
                outputs_e,
                g,
                loss,
                self.cfg.net.unique
            )
            delta_e, msg_size = self.re_compressor.c_dec(delta)
            self.last_res.r_communicate(msg_size)
            partial = self.zoos.backward(
                delta_e,
                us,
                self.cfg.net.unique
            )
        else:
            partial_1 = torch.autograd.grad(
                loss,
                outputs_e,
                retain_graph=True,
                only_inputs=True,
                allow_unused=False
            )[0].detach()
            partial, msg_size = self.re_compressor.c_dec(partial_1)
            self.last_res.r_communicate(msg_size)

        self.last_partial = partial
        return loss

    def step_onelayer(self, eta: Tensor) -> Tensor:
        inputs = self.last_inputs.detach()
        cecfg = self.cfg.adv.ce
        p_list = [
            -self.clients[m].model.layer_one_out.grad.detach()
            for m in range(self.cfg.net.n_client)
        ]
        eta.requires_grad_()
        eta.retain_grad()

        for _j in range(cecfg.n):
            adv_inp = inputs + eta
            adv_inp.clamp_(0., 1.)
            adv_list, _ = get_inputs_list(
                adv_inp,
                self.cfg.net.n_client,
                self.cfg.net.pad
            )

            hs = sum([self.clients[m].hamiltonian_func(
                adv_list[m], p_list[m]
            ) for m in range(self.cfg.net.n_client)])
            x_grad = torch.autograd.grad(
                hs, eta, only_inputs=True, retain_graph=False
            )[0].detach()

            eta = eta - x_grad.sign() * cecfg.sigma
            eta.clamp_(-cecfg.eps, cecfg.eps)
            eta = torch.clamp(inputs + eta, 0., 1.) - inputs
            eta = eta.detach()
            eta.requires_grad_()
            eta.retain_grad()

        adv_inp = inputs + eta
        adv_inp.clamp_(0., 1.)
        adv_list, _ = get_inputs_list(
            adv_inp,
            self.cfg.net.n_client,
            self.cfg.net.pad
        )

        sum([-self.clients[m].hamiltonian_func(
            adv_list[m], p_list[m]
        ) for m in range(self.cfg.net.n_client)]).backward()

        return adv_inp
