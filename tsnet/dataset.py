import random
import os
import math
import pandas as pd
from torch.utils.data import dataset, SubsetRandomSampler
from torchland.datasets.loader_builder import DataLoaderBuilder, DataLoader


class UCRDataLoaderBuilder(DataLoaderBuilder):
    def __init__(self, root_dir: str, dataset_name: str, batch_size: int, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = UCRDataset(root_dir, dataset_name, train=True)
        self.test_dataset = UCRDataset(root_dir, dataset_name, train=False)

        num_data = len(self.train_dataset)
        self.train_indices, self.validation_indices = self._split_train_val_indices(num_data)

    def make_train_dataloader(self) -> DataLoader:
        return self._dataloader_with_sampler(
            self.train_dataset, SubsetRandomSampler(self.train_indices))

    def make_validate_dataloader(self) -> DataLoader:
        return self._dataloader_with_sampler(
            self.train_dataset, SubsetRandomSampler(self.validation_indices))

    def make_test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True)

    @staticmethod
    def _split_train_val_indices(num_data: int):
        dataset_indices = list(range(num_data))  # 0, 1, ... n - 1
        random.shuffle(dataset_indices)
        num_train = math.floor(num_data * 0.9)
        return dataset_indices[:num_train], dataset_indices[num_train:]

    def _dataloader_with_sampler(self, ucr_dataset, sampler):
        return DataLoader(
            dataset=ucr_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=True)


class UCRDataset(dataset.Dataset):
    dataset_names = [
        'ACSF1',
        'Adiac',
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'ArrowHead',
        'BME',
        'Beef',
        'BeetleFly',
        'BirdChicken',
        'CBF',
        'Car',
        'Chinatown',
        'ChlorineConcentration',
        'CinCECGTorso',
        'Coffee',
        'Computers',
        'CricketX',
        'CricketY',
        'CricketZ',
        'Crop',
        # TODO: fill all
        'GunPoint',
    ]
    test_postfix = '_TEST'
    train_postfix = '_TRAIN'

    def __init__(self, root_dir: str, dataset_name: str, train: bool):
        assert dataset_name in self.dataset_names, f'dataset name should be one of: {self.dataset_names}'

        self.dataset_root = os.path.join(root_dir, dataset_name)

        # determine the postfix
        postfix = self._determine_postfix(train)
        filename = f'{dataset_name}{postfix}.tsv'

        self.dataset_path = os.path.join(self.dataset_root, filename)
        self.data = self._read_tsv(self.dataset_path)

    @classmethod
    def _determine_postfix(cls, train: bool):
        if train:
            postfix = cls.train_postfix
        else:
            postfix = cls.test_postfix
        return postfix

    @staticmethod
    def _read_tsv(fpath: str):
        return pd.read_csv(fpath, sep='\t', header=None)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx].to_numpy()
        class_label, timeseries = row[0], row[1:]
        return class_label, timeseries

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = UCRDataset(root_dir='/Users/dansuh/datasets/ucr-archive/UCRArchive_2018/',
                         dataset_name='GunPoint',
                         train=True)
