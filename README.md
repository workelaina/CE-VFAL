# CE-VFAL

CE-VFAL: A Novel Framework for Communication-Efficient Vertical Federated Adversarial Learning

[[IJCAI 2026](https://2026.ijcai.org/)]
[[Homepage](https://workelaina.github.io/CE-VFAL)]
[[PDF](https://workelaina.github.io/CE-VFAL/static/blob/CE-VFAL.pdf)]
[[Code](https://github.com/workelaina/CE-VFAL)]

## Abstract

Vertical Federated Learning (VFL) involves multiple participants collaborating to train machine learning models on distinct feature sets from the same data samples.
This training paradigm with distributed updating focuses on secure and efficient communication.
Nevertheless, the trained models exhibit heightened vulnerability to adversarial attacks during inference, which can provoke misclassification.
Adversarial Training (AT), which involves exposing models to intentionally crafted misleading examples during training, is widely regarded as the most effective method for enhancing model robustness.
However, the significant communication costs entailing such example generation within the VFL context pose an open challenge to developing a Vertical Federated Adversarial Learning (VFAL) framework.
To this end, we introduce a **C**ommunication-**E**fficient **V**ertical **F**ederated **A**dversarial **L**earning framework, named **CE-VFAL**.
The proposed framework incorporates the lazy propagation principle, confining most propagations to client models during adversarial updates, thereby *minimizing frequent client-server interactions*.
Moreover, CE-VFAL seamlessly integrates Zeroth Order Optimization (ZOO) into the communication phase, *effectively reducing communication load* by transmitting the loss difference derived from the raw and perturbed embeddings for multiple point estimation.
Furthermore, rigorous theoretical analysis demonstrates the sublinear convergence rate by containing the errors caused by multi-source approximate gradients.
Extensive experiments corroborate the robust performance while significantly reducing communication costs.

## Usage

### init

```shell
apt update
apt upgrade
apt install screen tree pciutils pkg-config
vim ~/.ssh/authorized_keys
vim ~/.bashrc
```

### env

#### env-docker

```shell
python -m pip install --upgrade pip wheel setuptools -i https://mirrors.jlu.edu.cn/pypi/simple
python -m pip install pillow matplotlib tqdm pandas scikit-learn scipy -i https://mirrors.jlu.edu.cn/pypi/simple
```

#### env-conda

```shell
conda create -y -n py312 python=3.12
conda activate py312
which python
# https://pytorch.org/
conda install pytorch torchvision torchaudio pillow matplotlib numpy tqdm pandas pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### check

```shell
nvidia-smi
py3smi
which python
python -c 'import torch;print(torch.cuda.is_available())'
lscpu
lsmem
lspci
```

### dev

```shell
python -m pip install seaborn
# python -m pip install transformers[torch]
# python -m pip install pytorch_transformers
# python -m pip install torchrec-nightly torchtext numba
```

### run

```shell
screen -R exp
cd exp
python train.py
```

## BibTeX

```bibtex
@inproceedings{CEVFAL2026TianxingMan,
  title     = {{CE-VFAL}: A Novel Framework for Communication-Efficient Vertical Federated Adversarial Learning},
  author    = {Tianxing Man and Jinjie Fang and Ganyu Wang and Yu Bai and Zhaogeng Liu and Bin Gu and Yi Chang},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on
               Artificial Intelligence, {IJCAI-26}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  year      = {2026},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2026/5737},
  url       = {https://doi.org/10.24963/ijcai.2026/5737},
}
```
