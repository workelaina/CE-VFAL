import os
import sys
import csv
import time
import random
import shutil
import numpy as np
import hashlib
from typing import Iterable
import matplotlib.pyplot as plt

import torch
from torch import Tensor

BIT_TO_MB = 2 ** 23
STR_LOSS = 'loss-'
STR_ACC = 'acc-'

DIRNAME = os.path.dirname(__file__)
CFGNAME = 'cfg.py'
CSVNAME = 'result.csv'
CFG_FILE = os.path.join(DIRNAME, CFGNAME)


def setup_seed(lucky_seed: str):
    # to make experimental results reproducible
    # keep same what? CPU? GPU? CUDA? torch?
    lucky_seed = int(hashlib.shake_256(
        lucky_seed.encode('utf8')
    ).hexdigest(4), 16)
    os.environ['PYTHONHASHSEED'] = str(lucky_seed)
    random.seed(lucky_seed)
    np.random.seed(lucky_seed)
    torch.manual_seed(lucky_seed)
    torch.cuda.manual_seed(lucky_seed)
    torch.cuda.manual_seed_all(lucky_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_last(root: str) -> str:
    for i in range(1, 99999):
        root_num = os.path.join(root, str(i))
        if not os.path.exists(root_num):
            return _l
        _l = root_num
    raise ValueError('too much dirs')


def find_next(root: str) -> str:
    for i in range(1, 99999):
        root_num = os.path.join(root, str(i))
        if not os.path.exists(root_num):
            os.makedirs(root_num, exist_ok=False)
            return root_num
    raise ValueError('too much dirs')


def get_inputs_list(
    inputs: Tensor,
    n_client: int,
    pad: bool
) -> Iterable[Iterable[Tensor]]:
    _c_shape_1 = (inputs.shape[-1] + n_client - 1) // n_client
    _c_n = (inputs.shape[-1] + _c_shape_1 - 1) // _c_shape_1
    if _c_n != n_client:
        raise RuntimeError("bcs of Tensor.chunk")
    _chunked = list(inputs.chunk(n_client, dim=-1))
    if not pad:
        return _chunked, _chunked
    _len_shape = len(inputs.shape)
    inputs_list = list()
    _l = 0
    for m in range(n_client):
        ans = torch.zeros_like(inputs)
        _r = _l + _chunked[m].shape[-1]
        if _len_shape == 2:
            ans[:, _l:_r] = _chunked[m]
        elif _len_shape == 3:
            ans[:, :, _l:_r] = _chunked[m]
        elif _len_shape == 4:
            ans[:, :, :, _l:_r] = _chunked[m]
        elif _len_shape == 5:
            ans[:, :, :, :, _l:_r] = _chunked[m]
        else:
            raise ValueError(inputs.shape)
        _l = _r
        inputs_list.append(ans)
    return inputs_list, _chunked


class EpochResult:
    def __init__(self, n_batch_prt: int = 100) -> None:
        self.n_batch_prt = n_batch_prt
        self.ce_size = 0
        self.r_ce_size = 0
        self.ce_num = 0
        self.r_ce_num = 0
        self.correct = 0
        self.running_loss = 0.0
        self.total_time = 0.0
        self.i_batch = 0
        self.len_data = 0

    def start_timer(self) -> None:
        self.time_start = time.time()

    def end_timer(self) -> None:
        self.total_time += time.time() - self.time_start
        assert self.time_start > 0
        self.time_start = -1

    def batch(self, output: Tensor, labels: Tensor, loss: Tensor) -> None:
        _, predicted = torch.max(output.data, 1)
        self.correct += (predicted == labels).sum().item()
        self.running_loss += loss.item()
        self.i_batch += 1
        self.len_data += labels.shape[0]
        if self.i_batch % self.n_batch_prt == 0:
            print('%.9f %.5fp' % (
                self.running_loss/self.len_data,
                self.correct*100/self.len_data
            ))

    def end_epoch(self) -> None:
        # print('[ce-1-epoch]', [
        #     self.ce_num, self.ce_size,
        #     self.r_ce_num, self.r_ce_size,
        #     self.i_batch, self.len_data
        # ])
        # sys.exit(0)
        r_mb = (self.ce_size + self.r_ce_size) / BIT_TO_MB
        if self.len_data > 0:
            r_loss = self.running_loss / self.len_data
            r_acc = self.correct * 100 / self.len_data
        else:
            r_loss = 0
            r_acc = 0
        return r_loss, r_acc, r_mb, self.total_time

    def communicate(self, x: int) -> int:
        self.ce_size += x
        self.ce_num += 1
        # return self.ce_size

    def r_communicate(self, x: int) -> int:
        self.r_ce_size += x
        self.r_ce_num += 1
        # return self.r_ce_size


def draw(
    lst: Iterable[float],
    title: str,
    pth: str
) -> None:
    plt.plot(range(len(lst)), lst)
    plt.title(title)
    # plt.legend()
    plt.grid()
    plt.savefig(pth)
    plt.clf()


class Logger:
    def __init__(
        self,
        field: Iterable[str],
        fmt: str,
        log_root: str
    ) -> None:
        self.field = field
        self.fmt = fmt
        self.data = dict()
        log_root = os.path.abspath(os.path.expanduser(log_root))
        self.root = find_next(log_root)
        shutil.copy2(CFG_FILE, os.path.join(self.root, CFGNAME))
        self.csv = os.path.join(self.root, CSVNAME)
        self.append(field, False)

    def append(self, row: Iterable, prt: bool = True) -> None:
        draw = prt
        with open(self.csv, 'a+') as f:
            write = csv.writer(f)
            write.writerow(row)
        if prt:
            print(self.fmt % tuple(row))
        if draw:
            _s = STR_LOSS+row[1]
            self.data.setdefault(_s, list())
            self.data[_s].append(row[2])
            _s = STR_ACC+row[1]
            self.data.setdefault(_s, list())
            self.data[_s].append(row[3])

    def draw(self, title: str) -> None:
        draw(
            self.data[title],
            title,
            os.path.join(self.root, title + '.png')
        )

    def draws(self) -> None:
        for i in self.data:
            self.draw(i)
