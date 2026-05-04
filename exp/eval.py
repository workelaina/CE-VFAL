import os
import sys

from cfg import cfg
from vflnet import VflNet

_ = cfg.device.split(':')

if len(_) > 1:
    assert _[0] == 'cuda'
    os.environ['CUDA_VISIBLE_DVICES'] = _[1]
else:
    assert _[0] == 'cpu'

vflnet = VflNet(cfg)

print()
print('Start Loading...')
print()

vflnet.load_model('./result/good_model/clean')

print('Start Testing')
print()

vflnet.test(is_eval=True)
