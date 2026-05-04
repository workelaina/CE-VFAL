import os
import sys

from cfg import cfg
from nvflnet import NVflNet
from vflnet import VflNet

_ = cfg.device.split(':')

if len(_) > 1:
    assert _[0] == 'cuda'
    os.environ['CUDA_VISIBLE_DVICES'] = _[1]
else:
    assert _[0] == 'cpu'

if cfg.net.n_party == 1:
    VflNet = NVflNet
vflnet = VflNet(cfg)

print()
print(cfg.adv.name)
print('Start Training...')
print()

vflnet.train()

print()
print('Finished Training')
print()

vflnet.test()
