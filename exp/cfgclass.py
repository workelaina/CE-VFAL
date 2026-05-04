from typing import Iterable
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class NetCfg:
    def __init__(
        self,
        n_party: int,
        msg_size: int,
        output_size: int,
        pad: bool,
        unique: bool,
        mu: float,
        q: int,
        u_type: Literal['uniform', 'normal', 'coordinate'],
        dataset: Literal['mnist', 'cifar10', 'fer', 'criteo'],
        name: str
    ) -> None:
        assert n_party >= 1
        assert u_type in ['uniform', 'normal', 'coordinate']
        assert dataset in ['mnist', 'cifar10', 'fer', 'criteo']
        self.n_party = n_party
        self.n_client = n_party - 1
        self.msg_size = msg_size
        self.output_size = output_size
        self.pad = pad
        self.unique = unique
        self.mu = mu
        self.q = q
        self.u_type = u_type
        self.dataset = dataset
        self.name = name


class OptimCfg:
    def __init__(
        self,
        name: Literal['adam', 'sgd'],
        lr: float,
        momentum: float,
        weight_decay: float,
        milestones: Iterable[int],
        gamma: float
    ) -> None:
        assert name in ['adam', 'sgd']
        self.name = name
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma


class PartyCfg:
    def __init__(
        self,
        optim: OptimCfg
    ) -> None:
        self.optim = optim


class AlgoCfg:
    def __init__(
        self,
        n: int,
        eps: float,
        sigma: float,
        random_start: bool
    ) -> None:
        self.n = n
        self.eps = eps
        self.sigma = sigma
        self.random_start = random_start


class CeCfg(AlgoCfg):
    def __init__(
        self,
        m: int,
        n: int,
        eps: float,
        sigma: float,
        random_start: bool,
        reg_cof: float,
        optim: OptimCfg
    ) -> None:
        super().__init__(n, eps, sigma, random_start)
        self.m = m
        self.reg_cof = reg_cof
        self.optim = optim


class PgdCfg(AlgoCfg):
    def __init__(
        self,
        n: int,
        eps: float,
        sigma: float,
        random_start: bool,
        norm: float
    ) -> None:
        super().__init__(n, eps, sigma, random_start)
        self.norm = norm


class CerCfg(AlgoCfg):
    def __init__(
        self,
        n: int,
        eps: float,
        sigma: float,
        random_start: bool
    ) -> None:
        super().__init__(n, eps, sigma, random_start)


class FreeatCfg(AlgoCfg):
    def __init__(
        self,
        n: int,
        eps: float,
        sigma: float,
        random_start: bool
    ) -> None:
        super().__init__(n, eps, sigma, random_start)


class FreelbCfg(AlgoCfg):
    def __init__(
        self,
        n: int,
        eps: float,
        sigma: float,
        random_start: bool
    ) -> None:
        super().__init__(n, eps, sigma, random_start)


class AdvCfg:
    def __init__(
        self,
        name: Literal['clean', 'ce', 'pgd', 'freeat', 'freelb', 'cer'],
        zo: bool,
        ce: CeCfg,
        pgd: PgdCfg,
        cer: CerCfg,
        freeat: FreeatCfg,
        freelb: FreelbCfg
    ) -> None:
        assert name in ['clean', 'ce', 'pgd', 'freeat', 'freelb', 'cer']
        self.name = name
        self.zo = zo
        self.ce = ce
        self.pgd = pgd
        self.cer = cer
        self.freeat = freeat
        self.freelb = freelb


class CwCfg(AlgoCfg):
    def __init__(
        self,
        m: int,
        n: int,
        eps: float,
        sigma: float,
        random_start: bool,
        kappa: float,
        prev: float,
        optim: OptimCfg
    ) -> None:
        super().__init__(n, eps, sigma, random_start)
        self.m = m
        self.kappa = kappa
        self.prev = prev
        self.optim = optim


class AaCfg(AlgoCfg):
    def __init__(
        self,
        n: int,
        eps: float,
        sigma: float,
        random_start: bool,
        norm: float,
        san: int,
        resc_schedule: int
    ) -> None:
        super().__init__(n, eps, sigma, random_start)
        self.norm = norm
        self.san = san
        self.resc_schedule = resc_schedule


class FgsmCfg(AlgoCfg):
    def __init__(
        self,
        n: int,
        eps: float,
        sigma: float,
        random_start: bool
    ) -> None:
        super().__init__(n, eps, sigma, random_start)


class AtkCfg:
    def __init__(
        self,
        zo: bool,
        pgd: PgdCfg,
        cer: CerCfg,
        cw: CwCfg,
        aa: AaCfg,
        fgsm: FgsmCfg
    ) -> None:
        self.zo = zo
        self.pgd = pgd
        self.cer = cer
        self.cw = cw
        self.aa = aa
        self.fgsm = fgsm


class ComCfg:
    def __init__(
        self,
        typ: Literal[None, 'scale', 'errfb', 'ieee754', 'topk', 'randk'],
        bit: int,
        re_typ: Literal[None, 'scale', 'errfb', 'ieee754', 'topk', 'randk'],
        re_bit: int
    ) -> None:
        assert typ in [None, 'scale', 'errfb', 'ieee754', 'topk', 'randk']
        assert bit > 0
        assert re_typ in [None, 'scale', 'errfb', 'ieee754', 'topk', 'randk']
        assert re_bit > 0
        self.typ = typ
        self.bit = bit
        self.re_typ = re_typ
        self.re_bit = re_bit


class LogCfg:
    def __init__(
        self,
        n_batch_prt: int,
        n_epoch_test: int,
        log_root: str
    ) -> None:
        self.n_batch_prt = n_batch_prt
        self.n_epoch_test = n_epoch_test
        self.log_root = log_root


class Cfg:
    def __init__(
        self,
        device: str,
        n_epoch: int,
        cpu_workers: int,
        batch_size: int,
        lucky_seed: str,
        log: LogCfg,
        net: NetCfg,
        com: ComCfg,
        adv: AdvCfg,
        atk: AtkCfg,
        client: PartyCfg,
        server: PartyCfg,
    ) -> None:
        self.device = device
        self.n_epoch = n_epoch
        self.cpu_workers = cpu_workers
        self.batch_size = batch_size
        self.lucky_seed = lucky_seed
        self.log = log
        self.net = net
        self.com = com
        self.adv = adv
        self.atk = atk
        self.client = client
        self.server = server
        self.model = client
