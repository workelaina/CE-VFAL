import numpy as np
import torch

from cfgclass import Cfg
from cfgclass import LogCfg, NetCfg, ComCfg, AdvCfg, AtkCfg, PartyCfg
from cfgclass import OptimCfg, PgdCfg, CwCfg, AaCfg, FgsmCfg, CerCfg
from cfgclass import CeCfg, FreeatCfg, FreelbCfg

EQ = '_cfg.py_eq_'
NEQ = '_cfg.py_neq_'

# ----------------------------------------------------------------------------

CPU = 8
# DataLoader(num_workers)
# num cores per GPU
# must be 0 on some old CPUs otherwise error

GPU = 0
DEBUG = True
ONLY_YOPO_CZO = False
# None EQ NEQ
VFL_LR_ALIGN = None
YOPO_LR_ALIGN = None

cfg = Cfg(
    device='cuda:%d' % GPU if torch.cuda.is_available() and GPU >= 0 else 'cpu',
    cpu_workers=CPU,
    n_epoch=999,
    batch_size=64,  # random
    lucky_seed='Lucky Seed!!',
    log=LogCfg(
        n_batch_prt=100,
        n_epoch_test=1,
        log_root='result'
    ),
    net=NetCfg(
        n_party=3,
        msg_size=128,  # random
        output_size=10,
        pad=True,
        unique=True,
        mu=0.01,  # random
        q=20,  # ac+
        u_type='uniform',
        dataset='mnist',
        name='resnet18'
    ),
    com=ComCfg(
        typ='randk',  # topk
        bit=64,  # 32
        re_typ=None,
        re_bit=8
    ),
    adv=AdvCfg(
        # m, n: num_iter, rob+
        name='pgd',
        zo=False,
        ce=CeCfg(
            m=5,
            n=10,
            eps=0.3,
            sigma=0.15,
            random_start=True,
            reg_cof=0,
            optim=OptimCfg(
                name='adam',
                lr=0.0001,
                momentum=0,
                weight_decay=5e-4,
                milestones=[10, 40],
                gamma=0.1
            )
        ),
        pgd=PgdCfg(
            n=40,
            eps=0.3,
            sigma=0.15,
            random_start=True,
            norm=np.inf
        ),
        cer=CerCfg(
            n=-1,  # free
            eps=0.3,
            sigma=-1.,  # free
            random_start=True
        ),
        freeat=FreeatCfg(
            n=8,
            eps=0.3,
            sigma=0.15,
            random_start=True
        ),
        freelb=FreelbCfg(
            n=40,
            eps=0.3,
            sigma=0.15,
            random_start=True
        )
    ),
    atk=AtkCfg(
        # must be the same within a column of a table
        # n: num_iter, atk+
        zo=False,
        pgd=PgdCfg(
            n=40,
            eps=0.3,
            sigma=0.15,
            random_start=True,
            norm=np.inf
        ),
        cer=CerCfg(
            n=-1,  # free
            eps=0.3,
            sigma=-1.,  # free
            random_start=True
        ),
        cw=CwCfg(
            m=20,
            n=100,
            eps=0.3,
            sigma=0.35,
            random_start=True,
            kappa=0,
            prev=1e10,
            optim=OptimCfg(
                name='adam',
                lr=0.1,
                momentum=0,
                weight_decay=0,
                milestones=[20],
                gamma=0.1
            )
        ),
        aa=AaCfg(
            n=40,
            eps=0.3,
            sigma=-1.,  # free
            random_start=True,  # free
            norm=np.inf,
            san=40,
            resc_schedule=True
        ),
        fgsm=FgsmCfg(
            n=-1,  # free
            eps=0.3,
            sigma=-1.,  # free
            random_start=True
        )
    ),
    client=PartyCfg(
        # better be the same within a column of a table
        optim=OptimCfg(
            name='adam',
            lr=0.0001,
            momentum=0,
            weight_decay=5e-4,
            milestones=[10, 40],
            gamma=0.1
        )
    ),
    server=PartyCfg(
        # better be the same within a column of a table
        optim=OptimCfg(
            name='adam',
            lr=0.0001,
            momentum=0,
            weight_decay=5e-4,
            milestones=[10, 40],
            gamma=0.1
        )
    )
)

assert not cfg.atk.zo

if DEBUG:
    cfg.atk.pgd.n = 2
    cfg.atk.cw.m = 999
    cfg.atk.cw.n = 2
    cfg.atk.aa.n = 2
    cfg.adv.ce.m = 2
    cfg.adv.ce.n = 2
    cfg.adv.pgd.n = 2
    cfg.adv.freeat.n = 2
    cfg.adv.freelb.n = 2
    # cfg.adv.ce.optim.lr *= 10
    # cfg.server.optim.lr *= 10
    # cfg.client.optim.lr *= 10
    cfg.n_epoch = 999
    cfg.log.n_epoch_test = 1
    ONLY_YOPO_CZO = False
    VFL_LR_ALIGN = None
    YOPO_LR_ALIGN = None
else:
    assert cfg.n_epoch >= 3
    assert cfg.net.q > 1
    assert cfg.net.pad == cfg.net.name.startswith('resnet')
    assert cfg.com.re_bit == 8
    # assert cfg.com.re_typ is None

if ONLY_YOPO_CZO:
    if cfg.adv.zo:
        assert cfg.adv.name == 'ce'
        assert cfg.com.typ == 'scale'
    else:
        assert cfg.adv.name != 'ce'
        assert cfg.com.typ is None

if VFL_LR_ALIGN is not None:
    if VFL_LR_ALIGN == EQ:
        assert cfg.server.optim.lr == cfg.client.optim.lr
    elif VFL_LR_ALIGN == NEQ:
        assert cfg.server.optim.lr == cfg.client.optim.lr * cfg.net.n_client
    else:
        raise ValueError(VFL_LR_ALIGN)

if YOPO_LR_ALIGN is not None:
    if YOPO_LR_ALIGN == EQ:
        assert cfg.client.optim.lr == cfg.adv.ce.optim.lr
    elif YOPO_LR_ALIGN == NEQ:
        assert cfg.client.optim.lr == cfg.adv.ce.optim.lr * cfg.adv.ce.n
    else:
        raise ValueError(YOPO_LR_ALIGN)
