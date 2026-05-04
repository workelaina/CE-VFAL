import os
import sys
import numpy as np
from typing import Iterable, Union, Callable

import torch
from torch import nn, Tensor

from cfgclass import Cfg, AlgoCfg
from dataset import get_dataset, make_data_loader
from function import gen_eta
from party import Client
from utils import Logger, EpochResult, find_next, find_last, setup_seed

STR_MODEL = 'model.checkpoint'


class NVflNet:
    def __init__(
        self,
        cfg: Cfg
    ) -> None:
        setup_seed(cfg.lucky_seed)
        self.cfg = cfg

        if cfg.net.name.startswith('mlp'):
            trainset, testset = get_dataset(cfg.net.dataset, flat=True)
        elif cfg.net.name.startswith('resnet'):
            trainset, testset = get_dataset(cfg.net.dataset, flat=False)
        else:
            raise ValueError(cfg.net.name)
        _shape = [cfg.batch_size] + list(trainset[0][0].shape)
        assert len(_shape) > 1
        self.trainloader = make_data_loader(
            trainset,
            batch_size=cfg.batch_size,
            num_workers=cfg.cpu_workers
        )
        self.testloader = make_data_loader(
            testset,
            batch_size=cfg.batch_size,
            num_workers=cfg.cpu_workers
        )

        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.BCELoss()
        self.last_loss = None
        self.last_partial = None
        self.logger = Logger(
            field=["epoch", "comm_round", "loss", "acc", "MB", "time"],
            fmt='%d %s: %.9f %.5fp %.2fMB %.2fs',
            log_root=cfg.log.log_root
        )

        # FreeAT
        self.global_noise_data = torch.zeros(_shape).to(cfg.device)
        self.init_model(_shape[1])

    def init_model(self, shape1: int) -> None:
        self.model = Client(self.cfg, shape1)
        self.lnkaa = self.model

    def save_model(self, x: Union[str, int] = None) -> str:
        if x is None:
            x = find_next(self.logger.root)
        if isinstance(x, int):
            x = os.path.join(self.logger.root, str(x))
        self.model.save(os.path.join(x, STR_MODEL))
        return x

    def load_model(self, x: Union[str, int] = None) -> str:
        if x is None:
            x = find_last(self.logger.root)
        if isinstance(x, int):
            x = os.path.join(self.logger.root, str(x))
        self.model.load(os.path.join(x, STR_MODEL))
        return x

    def gen_eta(
        self,
        shape: Iterable[int],
        algo: AlgoCfg
    ) -> Tensor:
        return gen_eta(
            self.cfg.device,
            shape,
            algo.eps,
            algo.random_start
        )

    def zero_grad(self) -> None:
        self.model.zero_grad()

    def step(self) -> None:
        self.model.step()

    def lr_step(self) -> None:
        self.model.lr_step()

    def zero_grad_ce(self) -> None:
        self.model.zero_grad_ce()

    def step_ce(self) -> None:
        self.model.step_ce()

    def backward(
        self,
        inputs: Iterable[str] = None,
        retain_graph: bool = None
    ) -> None:
        self.model.backward(
            self.last_outputs_fwd,
            None,
            inputs,
            retain_graph
        )

    def set_res(
        self,
        res: Union[EpochResult, int]
    ) -> None:
        if isinstance(res, int):
            res = EpochResult(res)
        self.last_res = res

    def start_timer(self) -> None:
        self.last_res.start_timer()

    def end_timer(self) -> None:
        self.last_res.end_timer()

    def end_epoch(
        self,
        _epoch: int,
        _name: str = ''
    ) -> None:
        row = self.last_res.end_epoch()
        row = [_epoch, _name] + list(row)
        self.logger.append(row)
        if _epoch >= 0:
            if _epoch % self.cfg.log.n_epoch_test == 0:
                self.test()
            self.logger.draws()

    def load_data(
        self,
        data: Tensor
    ) -> Iterable[Tensor]:
        inputs, labels = data
        inputs = inputs.to(self.cfg.device)
        labels = labels.to(self.cfg.device)
        self.last_inputs = inputs
        self.last_labels = labels
        return inputs, labels

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
        labels = self.last_labels

        output = self.model(inputs)

        if f is None:
            def f(_output: Tensor) -> Tensor:
                return self.loss_fn(_output, labels)

        loss = f(output)

        if append_log:
            self.last_res.batch(output, self.last_labels, loss)

        self.last_loss = loss
        self.last_outputs_fwd = loss
        return loss

    def pgd_loss(self, test: bool = False) -> Tensor:
        inputs = self.last_inputs

        if test:
            pgdcfg = self.cfg.atk.pgd
            zo = self.cfg.atk.zo
        else:
            pgdcfg = self.cfg.adv.pgd
            zo = self.cfg.adv.zo

        eta = self.gen_eta(inputs.shape, pgdcfg)
        # eta[:, 11:].zero_()

        for _i in range(pgdcfg.n):
            eta.requires_grad_()
            adv_inp = inputs + eta
            self.in2loss(adv_inp, zo=zo)

            x_grad = torch.autograd.grad(
                self.last_outputs_fwd,
                eta,
                grad_outputs=self.last_partial,
                only_inputs=True,
                retain_graph=False
            )[0].detach()

            adv_inp += x_grad.sign() * pgdcfg.sigma
            adv_inp.clamp_(0., 1.)
            eta = adv_inp - inputs
            eta.clamp_(-pgdcfg.eps, pgdcfg.eps)

        # eta[:, 11:].zero_()
        adv_inp = inputs + eta
        self.zero_grad()
        return self.in2loss(
            adv_inp,
            append_log=True,
            only_loss=test,
            zo=zo
        )

    def cer_loss(self, test: bool = False) -> Tensor:
        inputs = self.last_inputs

        if test:
            cercfg = self.cfg.atk.cer
            zo = self.cfg.atk.zo
        else:
            cercfg = self.cfg.adv.cer
            zo = self.cfg.adv.zo

        eta = torch.randn_like(inputs) * cercfg.eps
        # eta[:, 11:].zero_()
        adv_inp = inputs + eta
        return self.in2loss(
            adv_inp,
            append_log=True,
            only_loss=test,
            zo=zo
        )

    def cw_loss(self, targeted: bool = False) -> Tensor:
        inputs = self.last_inputs
        labels = self.last_labels
        cwcfg = self.cfg.atk.cw

        def g(_output: Tensor) -> Tensor:
            one_hot_labels = torch.eye(
                self.cfg.net.output_size
            ).to(self.cfg.device)[labels]

            f_i, _ = torch.max((1-one_hot_labels)*_output, dim=1)
            f_j = torch.masked_select(_output, one_hot_labels.bool())

            delta = f_i-f_j
            f_x = torch.clamp(delta if targeted else -delta, min=-cwcfg.kappa)
            return torch.sum(cwcfg.sigma*f_x)

        w = torch.zeros_like(inputs, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=cwcfg.optim.lr)
        prev = cwcfg.prev

        for _i in range(cwcfg.n):
            adv_inp = (nn.Tanh()(w) + 1) / 2
            loss1 = nn.MSELoss(reduction='sum')(adv_inp, inputs)
            self.in2loss(
                adv_inp,
                f=g,
                zo=self.cfg.atk.zo
            )
            cost = loss1 * (1-cwcfg.kappa) + self.last_loss * (1+cwcfg.kappa)
            optimizer.zero_grad()

            cost.backward()
            # loss1.backward(inputs=w, retain_graph=True)
            # self.last_outputs_fwd.backward(
            #     gradient=self.last_partial,
            #     inputs=w
            # )

            optimizer.step()

            # Early Stop when loss does not converge
            if _i % cwcfg.m == 0:
                if cost > prev:
                    print('Attack Stopped due to CONVERGENCE....')
                    break
                prev = cost

        adv_inp = (nn.Tanh()(w) + 1) / 2

        return self.in2loss(
            adv_inp,
            append_log=True,
            only_loss=True,
            zo=self.cfg.atk.zo
        )

    def aa_loss(self) -> Tensor:
        inputs = self.last_inputs
        labels = self.last_labels
        aacfg = self.cfg.atk.aa

        assert aacfg.norm == np.inf

        from autoattack import AutoAttack
        adversary = AutoAttack(
            self.lnkaa,
            eps=aacfg.eps,
            verbose=False,
            device=self.cfg.device,
            n_iter=aacfg.n,
            san=aacfg.san,
            sa_re=aacfg.resc_schedule,
            outputs_size=self.cfg.net.output_size
        )

        adv_inp = adversary.run_standard_evaluation(
            inputs, labels, bs=inputs.shape[0]
        )
        adv_inp.clamp_(0., 1.)
        eta = adv_inp - inputs
        # eta[:, 11:].zero_()
        eta.clamp_(-aacfg.eps, aacfg.eps)
        adv_inp = inputs + eta

        return self.in2loss(
            adv_inp,
            append_log=True,
            only_loss=True
        )

    def fgsm_loss(self) -> Tensor:
        inputs = self.last_inputs.detach()
        fcfg = self.cfg.atk.fgsm

        inputs.requires_grad_()
        self.in2loss(inputs, zo=self.cfg.atk.zo)

        x_grad = torch.autograd.grad(
            self.last_outputs_fwd,
            inputs,
            grad_outputs=self.last_partial,
            only_inputs=True,
            retain_graph=False
        )[0].detach()

        eta = x_grad.sign() * fcfg.eps
        # eta[:, 11:].zero_()
        adv_inp = inputs + eta
        adv_inp.clamp_(0., 1.)

        self.zero_grad()
        return self.in2loss(
            adv_inp,
            append_log=True,
            only_loss=True,
            zo=self.cfg.atk.zo
        )

    # def za_loss(self) -> Tensor:
    #     inputs = self.last_inputs.detach()
    #     labels = self.last_labels
    #     zacfg = self.cfg.atk.za

    #     from za import l2_attack
    #     use_log=True
    #     use_tanh=True
    #     targeted=True
    #     solver="adam"
    #     adv_inp = l2_attack(
    #         inputs,
    #         labels,
    #         self.lnkaa,
    #         targeted,
    #         use_log,
    #         use_tanh,
    #         solver,
    #         batch_size=inputs.shape[0],
    #         max_iter=zacfg.n,
    #         early_stop_iters=zacfg.n//10
    #     )

    #     return self.in2loss(
    #         adv_inp,
    #         append_log=True,
    #         only_loss=True
    #     )

    def test(self, is_eval: bool = False) -> None:
        self.save_model()
        # self.za_all()
        _inf = 9999999
        clean_res = EpochResult(_inf)
        pdg_res = EpochResult(_inf)
        cer_res = EpochResult(_inf)
        cw_res = EpochResult(_inf)
        aa_res = EpochResult(_inf)
        fgsm_res = EpochResult(_inf)
        # za_res = EpochResult(_inf)
        for _i, data in enumerate(self.testloader, 0):
            self.load_data(data)

            self.set_res(clean_res)
            self.start_timer()
            self.in2loss(append_log=True, only_loss=True)
            self.end_timer()

            self.set_res(pdg_res)
            self.start_timer()
            # self.pgd_loss(test=True)
            self.end_timer()

            self.set_res(cer_res)
            self.start_timer()
            # self.cer_loss(test=True)
            self.end_timer()

            self.set_res(cw_res)
            self.start_timer()
            # self.cw_loss()
            self.end_timer()

            self.set_res(aa_res)
            self.start_timer()
            # self.aa_loss()
            self.end_timer()

            self.set_res(fgsm_res)
            self.start_timer()
            self.fgsm_loss()
            self.end_timer()

            # self.set_res(za_res)
            # self.start_timer()
            # self.za_loss()
            # self.end_timer()

            if is_eval:
                self.set_res(clean_res)
                self.end_epoch(-1, 'test-clean')
                self.set_res(pdg_res)
                self.end_epoch(-2, 'test-pgd')
                self.set_res(cer_res)
                self.end_epoch(-3, 'test-cer')
                self.set_res(cw_res)
                self.end_epoch(-4, 'test-cw')
                self.set_res(aa_res)
                self.end_epoch(-5, 'test-autoatk')
                self.set_res(fgsm_res)
                self.end_epoch(-6, 'test-fgsm')
                # self.set_res(za_res)
                # self.end_epoch(-7, 'test-zooatk')

        self.set_res(clean_res)
        self.end_epoch(-1, 'test-clean')
        self.set_res(pdg_res)
        self.end_epoch(-2, 'test-pgd')
        self.set_res(cer_res)
        self.end_epoch(-3, 'test-cer')
        self.set_res(cw_res)
        self.end_epoch(-4, 'test-cw')
        self.set_res(aa_res)
        self.end_epoch(-5, 'test-autoatk')
        self.set_res(fgsm_res)
        self.end_epoch(-6, 'test-fgsm')
        # self.set_res(za_res)
        # self.end_epoch(-7, 'test-zooatk')

    def clean_epoch(self) -> None:
        self.in2loss(append_log=True, zo=self.cfg.adv.zo)
        self.zero_grad()
        self.backward()
        self.step()

    def pgd_epoch(self) -> None:
        self.pgd_loss(test=False)
        self.zero_grad()
        self.backward()
        self.step()

    def cer_epoch(self) -> None:
        self.cer_loss(test=False)
        self.zero_grad()
        self.backward()
        self.step()

    def freeat_epoch(self) -> None:
        inputs = self.last_inputs
        atcfg = self.cfg.adv.freeat

        for _i in range(atcfg.n):
            eta = self.global_noise_data[:inputs.shape[0]]
            eta.requires_grad_()
            adv_inp = inputs + eta
            adv_inp.clamp_(0., 1.)
            self.in2loss(
                adv_inp,
                append_log=True if _i == atcfg.n - 1 else False,
                zo=self.cfg.adv.zo
            )

            self.zero_grad()
            x_grad = torch.autograd.grad(
                self.last_outputs_fwd,
                eta,
                grad_outputs=self.last_partial,
                only_inputs=True,
                retain_graph=True
            )[0].detach()
            self.backward()
            self.step()

            pert = atcfg.sigma * x_grad.sign()
            self.global_noise_data[:inputs.shape[0]] += pert
            self.global_noise_data.clamp_(-atcfg.eps, atcfg.eps)

    def freelb_epoch(self) -> None:
        inputs = self.last_inputs.detach()
        labels = self.last_labels
        inputs.requires_grad_()
        lbcfg = self.cfg.adv.freelb

        def g(_output: Tensor) -> Tensor:
            return self.loss_fn(_output, labels) / lbcfg.n

        def project(x: Tensor, eps: Tensor) -> Tensor:
            # project X on the ball of radius eps supposing first dim is batch
            dims = list(range(1, x.dim()))
            norms = torch.sqrt(torch.sum(x*x, dim=dims, keepdim=True))
            return torch.min(norms.new_ones(norms.shape), eps/norms) * x

        n_eta = inputs[0].numel()
        # What is 21128*768?
        eta = self.gen_eta(
            inputs.shape,
            lbcfg
        ) / np.sqrt(n_eta)

        self.zero_grad()
        for _i in range(lbcfg.n):
            adv_inp = inputs + eta
            self.in2loss(
                adv_inp,
                f=g,
                append_log=True if _i == lbcfg.n - 1 else False,
                zo=self.cfg.adv.zo
            )
            self.backward(retain_graph=True)

            self.last_outputs_fwd.backward(
                gradient=self.last_partial,
                retain_graph=False,
                inputs=inputs
            )
            x_grad = inputs.grad.detach()

            pert = lbcfg.sigma * x_grad / torch.norm(x_grad)
            eta = project(eta + pert, lbcfg.eps)

        self.step()

    def step_onelayer(self, eta: Tensor) -> Tensor:
        inputs = self.last_inputs.detach()
        cecfg = self.cfg.adv.ce
        p = -self.model.model.layer_one_out.grad.detach()
        eta.requires_grad_()
        eta.retain_grad()

        for _j in range(cecfg.n):
            adv_inp = inputs + eta
            adv_inp.clamp_(0., 1.)

            H = self.model.hamiltonian_func(adv_inp, p)
            x_grad = torch.autograd.grad(
                H, adv_inp, only_inputs=True, retain_graph=False
            )[0].detach()

            eta = eta - x_grad.sign() * cecfg.sigma
            eta.clamp_(-cecfg.eps, cecfg.eps)
            eta = torch.clamp(inputs + eta, 0., 1.) - inputs
            eta = eta.detach()
            eta.requires_grad_()
            eta.retain_grad()

        adv_inp = inputs + eta
        adv_inp.clamp_(0., 1.)

        loss = -self.model.hamiltonian_func(adv_inp, p)
        loss.backward()

        return adv_inp

    def ce_epoch(self) -> None:
        inputs = self.last_inputs
        cecfg = self.cfg.adv.ce
        eta = self.gen_eta(inputs.shape, cecfg)
        self.zero_grad_ce()
        adv_inp = inputs + eta.detach()
        for _i in range(cecfg.m):
            self.in2loss(
                adv_inp,
                append_log=True if _i == cecfg.m - 1 else False,
                zo=self.cfg.adv.zo
            )
            self.backward(inputs=['other_layers', 'layer_one_out'])
            adv_inp = self.step_onelayer(eta.detach()).detach()
        self.step_ce()

    def train_epoch(self) -> None:
        if self.cfg.adv.name == 'ce':
            return self.ce_epoch()
        elif self.cfg.adv.name == 'clean':
            return self.clean_epoch()
        elif self.cfg.adv.name == 'pgd':
            return self.pgd_epoch()
        elif self.cfg.adv.name == 'cer':
            return self.cer_epoch()
        elif self.cfg.adv.name == 'freeat':
            return self.freeat_epoch()
        elif self.cfg.adv.name == 'freelb':
            return self.freelb_epoch()
        else:
            raise ValueError(self.cfg.adv.name)

    def train(self) -> None:
        for _epoch in range(self.cfg.n_epoch):
            self.set_res(self.cfg.log.n_batch_prt)
            self.start_timer()
            for data in self.trainloader:
                self.load_data(data)
                self.train_epoch()
            self.lr_step()
            self.end_timer()
            self.end_epoch(_epoch, 'train-' + self.cfg.adv.name)
