import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset
import torch.utils.data
from torchvision import datasets, transforms

DATA_DIR = '../data/'


class CriteoDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_criteo_data(file_path):
    data = pd.read_csv(file_path)
    labels = data.iloc[:, -1].values
    features = data.iloc[:, :-1]

    numeric_features = features.select_dtypes(
        include=['int64', 'float64']
    ).columns
    categorical_features = features.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        # ('num', MinMaxScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    features = preprocessor.fit_transform(features)

    return features, labels


class Td(Dataset):
    def __init__(self, df: DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        labels = torch.tensor(
            self.df.iloc[index].iloc[0],
            dtype=torch.int64
        )
        inputs = torch.tensor(
            self.df.iloc[index].iloc[1:],
            dtype=torch.float32
        )
        return inputs, labels


def get_dataset(name: str, flat: bool = False):
    if name == 'mnist':
        _l = [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]
        if flat:
            _l.append(transforms.Lambda(torch.flatten))
        trainset = datasets.MNIST(
            DATA_DIR, download=True, train=True,
            transform=transforms.Compose(_l)
        )
        testset = datasets.MNIST(
            DATA_DIR, download=True, train=False,
            transform=transforms.Compose(_l)
        )
    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.4914, 0.4822, 0.4465),
            #     (0.2023, 0.1994, 0.2010)
            # )
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.4914, 0.4822, 0.4465),
            #     (0.2023, 0.1994, 0.2010)
            # )
        ])
        trainset = datasets.CIFAR10(
            DATA_DIR, download=True, train=True,
            transform=transform_train
        )
        testset = datasets.CIFAR10(
            DATA_DIR, download=True, train=False,
            transform=transform_test
        )
    elif name == 'fer':
        _l = [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]
        dataset = datasets.FER2013(
            DATA_DIR, download=True, train=True,
            transform=transforms.Compose(_l)
        )
        trsz = int(len(dataset) * .7 + 0.5)
        tesz = len(dataset) - trsz
        trainset, testset = torch.utils.data.random_split(
            dataset,
            [trsz, tesz]
        )
    elif name == 'criteo':
        file_path = os.path.join(DATA_DIR, 'criteo.csv')
        features, labels = load_criteo_data(file_path)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=0.3, stratify=labels
        )
        trainset = CriteoDataset(X_train, y_train)
        testset = CriteoDataset(X_test, y_test)

    else:
        raise ValueError(name)
    return trainset, testset


def make_data_loader(
    dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False
):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return data_loader
