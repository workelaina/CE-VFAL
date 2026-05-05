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

## BibTeX

```bibtex
@inproceedings{CEVFAL2026TianxingMan,
}
```
