import sys
from typing import Iterable, Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np

import torch
from torch import Tensor

TORCH_FLOAT = [torch.float16, torch.float32, torch.float64]


def tensor_size(x: Tensor, bit: int = None) -> int:
    n_element = torch.numel(x)
    if x.dtype in TORCH_FLOAT or bit is None:
        bit = x.element_size() * 8
    return n_element * bit


def tensors_size(y: Iterable[Tensor], bit: int = None) -> int:
    ans = 0
    for x in y:
        ans += tensor_size(x, bit)
    return ans


class Compressor:
    def __init__(
        self,
        device: str,
        typ: Literal[None, 'scale', 'errfb', 'ieee754', 'topk', 'randk'],
        bit: int,
    ) -> None:
        self.device = device
        self.typ = typ
        self.bit = bit

        if typ in ['scale', 'errfb']:
            if bit < 32:
                self.dtype = torch.int
            elif bit < 64:
                self.dtype = torch.int64
            elif bit == 64:
                self.dtype = torch.uint64
            else:
                raise ValueError(bit)
            self.num = 2**bit
            if typ == 'errfb':
                self.errfb_old = None
        elif typ in ['topk', 'randk']:
            self.dtype = None
        elif typ == 'ieee754':
            if bit == 16:
                self.dtype = torch.float16
            elif bit == 32:
                self.dtype = torch.float32
            elif bit == 64:
                self.dtype = torch.float64
            else:
                raise ValueError(bit)
        else:
            assert typ is None

    def c_s(self, x: Tensor) -> Iterable[Tensor]:
        if self.typ == 'errfb':
            if self.errfb_old is not None:
                x -= self.errfb_old[:x.shape[0]]
        low = x.min()
        high = x.max()
        boundaries = torch.linspace(low, high, self.num).to(self.device)
        compressed = torch.bucketize(x, boundaries).to(self.dtype)
        return compressed, low, high

    def dec_s(self, y: Iterable[Tensor]) -> Tensor:
        compressed, low, high = y
        rx = low + compressed * (high - low) / self.num
        if self.typ == 'errfb':
            if self.errfb_old is not None:
                rx += self.errfb_old[:rx.shape[0]]
                self.errfb_old[:rx.shape[0]] = rx.detach()
            else:
                self.errfb_old = rx.detach()
        return rx

    def c_k(self, x: Tensor) -> Tensor:
        topk_values, topk_indices = x.topk(self.bit)
        out = torch.zeros_like(x)

        if len(out.shape) == 1:
            out[topk_indices] = topk_values
        elif len(out.shape) == 2:
            for i in range(out.shape[0]):
                out[i][topk_indices[i]] = topk_values[i]
        else:
            raise ValueError(out.shape)
        return out

    def dec_k(self, y: Tensor) -> Tensor:
        return y

    def c_dec(self, x: Tensor) -> Tuple[Tensor, int]:
        if self.typ is None:
            y = x.detach()
            size = tensor_size(y)
            rx = y
        elif self.typ in ['scale', 'errfb']:
            y = self.c_s(x.detach())
            size = tensors_size(y, self.bit)
            rx = self.dec_s(y)
        elif self.typ in ['topk', 'randk']:
            if self.typ == 'topk':
                _v, _k = x.topk(self.bit)

            rx = torch.zeros_like(x.detach())

            if len(x.shape) == 1:
                if self.typ == 'topk':
                    rx[_k] = _v
                else:
                    _k = np.random.choice(x.shape[-1], self.bit)
                    rx[_k] = x[_k]
                size = self.bit * 32 * 2

            elif len(x.shape) == 2:
                if self.typ == 'topk':
                    for _i in range(x.shape[0]):
                        rx[_i][_k[_i]] = _v[_i]
                else:
                    for _i in range(x.shape[0]):
                        _k = np.random.choice(x.shape[-1], self.bit)
                        rx[_i][_k] = x[_i][_k]
                size = self.bit * 32 * 2 * rx.shape[0]

            else:
                raise ValueError(len(x.shape))

        elif self.typ == 'ieee754':
            y = x.detach().to(self.dtype)
            size = tensor_size(y)
            rx = y
        else:
            raise ValueError(self.typ)
        return rx, size
