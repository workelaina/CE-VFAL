import sys
from typing import Iterable, Union

import torch
from torch import Tensor

from cfgclass import Cfg, PartyCfg
from function import Hamiltonian
from network import Client_L1, Server_L2, myResNet


class Party:
    def __init__(
        self,
        cfg: PartyCfg,
        model: Client_L1,
    ) -> None:
        self.model = model
        if cfg.optim.name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.optim.lr,
                weight_decay=cfg.optim.weight_decay
            )
        elif cfg.optim.name == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.optim.lr,
                momentum=cfg.optim.momentum,
                weight_decay=cfg.optim.weight_decay
            )
        else:
            raise ValueError(cfg.optim.name)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=cfg.optim.milestones,
            gamma=cfg.optim.gamma
        )

    def __call__(self, x: Tensor) -> Tensor:
        return self.model(x)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()

    def lr_step(self) -> None:
        self.lr_scheduler.step()

    def get_inputs(
        self,
        inputs: Union[Iterable[str], str] = None
    ) -> Iterable[Tensor]:
        if inputs is None or inputs == 'model':
            return list(self.model.parameters())
        if isinstance(inputs, str) or len(inputs) != 1:
            raise ValueError(inputs)
        inputs = inputs[0]
        if inputs is None or inputs == 'model':
            return list(self.model.parameters())
        raise ValueError(inputs)

    def backward(
        self,
        outputs: Tensor,
        partial: Tensor = None,
        inputs: Iterable[str] = None,
        retain_graph: bool = None
    ) -> None:
        outputs.backward(
            gradient=partial,
            retain_graph=retain_graph,
            inputs=self.get_inputs(inputs)
        )

    def save(self, pth: str) -> None:
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict()
        }, pth)

    def load(self, pth: str) -> None:
        checkpoint = torch.load(pth)
        cp_m = checkpoint['model']
        try:
            if 'fc1.weight' not in cp_m:
                cp_m['fc1.weight'] = cp_m['conv1.weight']
                cp_m['fc1.bias'] = cp_m['conv1.bias']
        except KeyError:
            pass
        self.model.load_state_dict(checkpoint['model'],strict=False)

        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


class Client(Party):
    def __init__(self, cfg: Cfg, shape1: int = None) -> None:
        if cfg.net.n_client:
            output_size = cfg.net.msg_size
        else:
            output_size = cfg.net.output_size
        if cfg.net.name.startswith('mlp'):
            if cfg.net.n_client and not cfg.net.pad:
                if shape1 % cfg.net.n_client:
                    raise RuntimeError('%d to %d chunk')
                shape1 = shape1 // cfg.net.n_client
            model = Client_L1(
                input_size=shape1,
                output_size=output_size
            ).to(cfg.device)
        elif cfg.net.name.startswith('resnet'):
            assert cfg.net.pad
            model = myResNet(
                resnet_num=int(cfg.net.name[6:]),
                in_channels=shape1,
                output_size=output_size
            ).to(cfg.device)
        else:
            raise ValueError(cfg.net.name)
        super().__init__(cfg.client, model)

        self.other_layers = model.other_layers
        self.layer_one = model.layer_one
        self.conv1 = model.conv1

        if cfg.client.optim.name == 'adam':
            self.other_optimizer = torch.optim.Adam(
                self.other_layers.parameters(),
                lr=cfg.client.optim.lr,
                weight_decay=cfg.client.optim.weight_decay
            )
        elif cfg.client.optim.name == 'sgd':
            self.other_optimizer = torch.optim.SGD(
                self.other_layers.parameters(),
                lr=cfg.client.optim.lr,
                momentum=cfg.client.optim.momentum,
                weight_decay=cfg.client.optim.weight_decay
            )
        else:
            raise ValueError(cfg.client.optim.name)

        self.other_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.other_optimizer,
            milestones=cfg.client.optim.milestones,
            gamma=cfg.client.optim.gamma,
        )

        cecfg = cfg.adv.ce

        if cecfg.optim.name == 'adam':
            self.one_optimizer = torch.optim.Adam(
                self.layer_one.parameters(),
                lr=cecfg.optim.lr,
                weight_decay=cecfg.optim.weight_decay
            )
        elif cecfg.optim.name == 'sgd':
            self.one_optimizer = torch.optim.SGD(
                self.layer_one.parameters(),
                lr=cecfg.optim.lr,
                momentum=cecfg.optim.momentum,
                weight_decay=cecfg.optim.weight_decay
            )
        else:
            raise ValueError(cfg.client.optim.name)

        self.one_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.one_optimizer,
            milestones=cecfg.optim.milestones,
            gamma=cecfg.optim.gamma,
        )

        self.hamiltonian_func = Hamiltonian(
            self.layer_one,
            cecfg.reg_cof
        )

    def get_inputs(
        self,
        inputs: Iterable[str] = None
    ) -> Iterable[Tensor]:
        if inputs is None:
            return list(self.model.parameters())
        if isinstance(inputs, str):
            inputs = [inputs]
        ans = list()
        for i in inputs:
            if i in ['model']:
                ans += list(self.model.parameters())
            # elif i in ['inp', 'input', 'x', 'eta', 'adv_inp']:
            #     ans.append(self.model.adv_inp)
            elif i in ['layer_one_out', 'y']:
                ans.append(self.model.layer_one_out)
            elif i in ['other_layers']:
                ans += list(self.other_layers.parameters())
            elif i in ['layer_one']:
                # ans += list(self.layer_one.parameters())
                raise ValueError(i)
            else:
                raise ValueError(i)
        return ans

    def zero_grad_ce(self) -> None:
        self.other_optimizer.zero_grad()
        self.one_optimizer.zero_grad()
        try:
            self.model.layer_one_out.grad.zero_()
        except AttributeError:
            pass
        except RuntimeError:
            pass

    def step_ce(self) -> None:
        self.other_optimizer.step()
        self.one_optimizer.step()

    def lr_step(self) -> None:
        super().lr_step()
        self.other_lr_scheduler.step()
        self.one_lr_scheduler.step()

    # def save(self, pth: str) -> None:
    #     torch.save({
    #         'model': self.model.state_dict(),
    #         'optimizer': self.optimizer.state_dict(),
    #         'lr_scheduler': self.lr_scheduler.state_dict(),
    #         'other_optimizer': self.other_optimizer.state_dict(),
    #         'other_lr_scheduler': self.other_lr_scheduler.state_dict(),
    #         'one_optimizer': self.one_optimizer.state_dict(),
    #         'one_lr_scheduler': self.one_lr_scheduler.state_dict()
    #     }, pth)

    # def load(self, pth: str) -> None:
    #     cp = torch.load(pth)
    #     self.model.load_state_dict(cp['model'])
    #     self.optimizer.load_state_dict(cp['optimizer'])
    #     self.lr_scheduler.load_state_dict(cp['lr_scheduler'])
    #     self.other_optimizer.load_state_dict(cp['other_optimizer'])
    #     self.other_lr_scheduler.load_state_dict(cp['other_lr_scheduler'])
    #     self.one_optimizer.load_state_dict(cp['one_optimizer'])
    #     self.one_lr_scheduler.load_state_dict(cp['one_lr_scheduler'])


class Server(Party):
    def __init__(
        self,
        cfg: Cfg,
        output_size: int,
    ) -> None:
        model = Server_L2(
            input_size=cfg.net.msg_size*cfg.net.n_client,
            output_size=output_size
        ).to(cfg.device)
        super().__init__(cfg.server, model)
