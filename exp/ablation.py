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

assert cfg.net.name == 'resnet18'
assert cfg.net.dataset == 'mnist'
assert cfg.net.q == 20
assert cfg.adv.zo
assert cfg.adv.name == 'ce'
assert cfg.adv.ce.m == 5
assert cfg.adv.ce.n == 10
assert cfg.com.typ == 'scale'
assert cfg.com.bit == 8
assert cfg.com.re_typ is None

vflnet = VflNet(cfg)

# print()
# print(cfg.adv.name, 'zozo')
# print('Start Training...')
# print()

# vflnet.train()

# print()
# print('Finished Training')
# print()

# vflnet.test()

# cfg.adv.zo = False
# vflnet = VflNet(cfg)

# print()
# print(cfg.adv.name, 'fofo')
# print('Start Training...')
# print()

# vflnet.train()

# print()
# print('Finished Training')
# print()

# vflnet.test()

cfg.adv.zo = True

for m in [3, 8]:
    cfg.adv.ce.m = m
    for n in range(31):
        cfg.adv.ce.n = n
        vflnet = VflNet(cfg)

        print()
        print(cfg.adv.name, m, n)
        print('Start Training...')
        print()

        vflnet.train()

        print()
        print('Finished Training')
        print()

        vflnet.test()

# cfg.adv.ce.m = 5
# cfg.adv.ce.n = 10

# for b in [1, 2, 4, 8, 16, 32]:
#     cfg.com.bit = b
#     vflnet = VflNet(cfg)

#     print()
#     print(cfg.adv.name, 'fc', b)
#     print('Start Training...')
#     print()

#     vflnet.train()

#     print()
#     print('Finished Training')
#     print()

#     vflnet.test()

# cfg.com.typ = None
# cfg.com.re_typ = 'scale'

# for b in [1, 2, 4, 8, 16, 32]:
#     cfg.com.re_bit = b
#     vflnet = VflNet(cfg)

#     print()
#     print(cfg.adv.name, 'bc', b)
#     print('Start Training...')
#     print()

#     vflnet.train()

#     print()
#     print('Finished Training')
#     print()

#     vflnet.test()

# cfg.com.typ = None
# cfg.com.re_typ = None

# vflnet = VflNet(cfg)

# print()
# print(cfg.adv.name, 'not c')
# print('Start Training...')
# print()

# vflnet.train()

# print()
# print('Finished Training')
# print()

# vflnet.test()
