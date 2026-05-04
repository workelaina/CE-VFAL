import sys
from typing import Iterable, Callable
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
from torch import Tensor
import torch.nn.functional as F

# _APPOINTED_K = 0
# _NEXT_K = _APPOINTED_K
# _L_K = 2 ** 32
# _MX_K = 2 ** 16


class Zoo:
    def __init__(
        self,
        device: str,
        mu: float,
        u_type: Literal['uniform', 'normal', 'coordinate'],
        shape: Iterable[int],
        coordinate_l: int = 0
    ) -> None:
        self.device = device
        self.mu = mu
        self.u_type = u_type
        self.shape = shape
        self.coordinate_l = coordinate_l
        # This implementation only works for
        # the clients model has same output dimension
        # otherwise the Phi should implement in backward
        if self.u_type == 'uniform':
            self.phi = self.shape[-1]
        elif self.u_type == 'normal':
            self.phi = 1.
        elif u_type == 'coordinate':
            self.phi = .5
        else:
            raise ValueError('no selected u_type %s' % self.u_type)
        self.u = self.get_u(unique=True)

    def get_u(self, unique: bool = True) -> Tensor:
        if not unique:
            return self.u
        # global _NEXT_K
        # torch.manual_seed(_NEXT_K)
        # _NEXT_K = int(torch.randn(1).abs_() * _L_K) % _MX_K
        if self.u_type == 'uniform':
            u = torch.randn(self.shape).to(self.device)
            u = F.normalize(u, dim=-1)
        elif self.u_type == 'normal':
            u = torch.randn(self.shape).to(self.device)
        elif self.u_type == 'coordinate':
            u = torch.ones(self.shape).to(self.device)
        else:
            raise ValueError('no selected u_type %s' % self.u_type)
        return u

    def forward(self, x: Tensor, unique: bool = True) -> Iterable[Tensor]:
        if self.u_type == 'coordinate':
            return Tensor([
                x + self.mu, x - self.mu
            ]).to(self.device), self.u[:x.shape[0]]
        # perturb the gradient
        u = self.get_u(unique)[:x.shape[0]]
        return x + self.mu * u, u

    def backward(
        self,
        delta: Tensor,
        u: Tensor = None,
        unique: bool = None
    ) -> Tensor:
        if u is None:
            assert not unique
            u = self.u
        partial = self.phi / self.mu * delta.view(-1, 1) * u
        return partial

    def zo_delta(
        self,
        x: Tensor,
        f: Callable[[Tensor,], Tensor],
        loss: Tensor,
        unique: bool = True
    ) -> Iterable[Tensor]:
        if self.u_type == 'coordinate':
            delta = f(x + self.mu) - f(x - self.mu)
            return delta, self.u[:x.shape[0]]
        p, u = self.forward(x, unique)
        delta = f(p) - loss
        return delta, u


class Zoos:
    def __init__(
        self,
        q: int,
        device: str,
        mu: float,
        u_type: Literal['uniform', 'normal', 'coordinate'],
        shape: Iterable[int],
        coordinate_l: int = 0
    ):
        self.device = device
        self.q = q
        self.mu = mu
        self.u_type = u_type
        self.zos = [Zoo(
            device, mu, u_type, shape, coordinate_l
        ) for _ in range(q)]

    def forward(
        self,
        x: Tensor,
        unique: bool = True
    ) -> Iterable[Iterable[Tensor]]:
        return [zo.forward(x, unique) for zo in self.zos]

    def backward(
        self,
        deltas: Iterable[Tensor],
        us: Iterable[Tensor] = None,
        unique: bool = None
    ) -> Tensor:
        if us is None:
            assert not unique
            us = [zo.u for zo in self.zos]
        partial = 0
        for i, zo in enumerate(self.zos):
            partial += zo.backward(deltas[i], us[i], unique)
        return partial / self.q

    def zo_delta(
        self,
        x: Tensor,
        f: Callable[[Tensor,], Tensor],
        loss: Tensor,
        unique: bool = None
    ) -> Iterable[Iterable[Tensor]]:
        deltas = list()
        us = list()
        for zo in self.zos:
            delta, u = zo.zo_delta(x, f, loss, unique)
            us.append(u)
            deltas.append(delta)

        return Tensor(deltas).detach().to(self.device), us
