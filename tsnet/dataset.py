import random
import os
import math
import pandas as pd
import numpy as np
import scipy.signal as signal
from torch.utils.data import dataset, SubsetRandomSampler
from torchland.datasets.loader_builder import DataLoaderBuilder, DataLoader


class UCRDataLoaderBuilder(DataLoaderBuilder):
    def __init__(self, root_dir: str, dataset_name: str,
                 batch_size: int, num_workers=4,
                 **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = UCRDataset(root_dir, dataset_name, train=True, **kwargs)
        self.test_dataset = UCRDataset(root_dir, dataset_name, train=False, **kwargs)

        num_data = len(self.train_dataset)
        self.train_indices, self.validation_indices = self._split_train_val_indices(num_data)

        self.num_train = len(self.train_indices)
        self.num_val = len(self.validation_indices)
        self.num_test = len(self.test_dataset)

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
            pin_memory=True,
        )


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

    def __init__(self, root_dir: str, dataset_name: str, train: bool, sample_length: int):
        assert dataset_name in self.dataset_names, f'dataset name should be one of: {self.dataset_names}'

        self.dataset_root = os.path.join(root_dir, dataset_name)

        # determine the postfix
        postfix = self._determine_postfix(train)
        filename = f'{dataset_name}{postfix}.tsv'

        self.dataset_path = os.path.join(self.dataset_root, filename)
        self.data = self._read_tsv(self.dataset_path)
        if len(self.data) == 0:
            raise ValueError('Number of data cannot be 0.')

        _, sample_ts = self._get_row(idx=0)
        self.data_sample_length = len(sample_ts)
        self.sample_length = sample_length
        if self.data_sample_length <= self.sample_length:
            raise ValueError(
                f'Data samples have length {self.data_sample_length}, '
                f'while anchor length require: {self.sample_length}')

        self.resamp_ratio_low = 0.8
        self.resamp_ratio_high = 1.2
        self.cut_ratio_low = 0.6
        self.cut_ratio_high = 0.9

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
        class_label, sample = self._get_row(idx)
        anchor = self._random_cut_size(sample, size=self.sample_length)

        positive_samp = self.random_resamp_cut(anchor)
        # positive_samp = self._random_cut_size(sample, size=self.sample_length - 20)

        # mine negative sample
        negative_idx = self._select_idx_except(except_idx=class_label)  # TODO: debug
        _, neg_sample = self._get_row(negative_idx)
        negative_samp = self.random_resamp_cut(neg_sample)
        # negative_samp = self._random_cut_size(neg_sample, size=self.sample_length - 20)

        # specify the type and add an axis to the first dimension
        anchor = self._add_axis(self._to_float32(anchor))
        positive_samp = self._add_axis(self._to_float32(positive_samp))
        negative_samp = self._add_axis(self._to_float32(negative_samp))
        return class_label, anchor, positive_samp, negative_samp

    def random_resamp_cut(self, sample: np.ndarray):
        return self._random_resamp_cut(
            sample,
            resamp_ratio_low=self.resamp_ratio_low,
            resamp_ratio_high=self.resamp_ratio_high,
            cut_ratio_low=self.cut_ratio_low,
            cut_ratio_high=self.cut_ratio_high,
        )

    @staticmethod
    def _random_cut_size(sample: np.ndarray, size: int):
        num_samps = len(sample)
        start_pos = np.random.randint(low=0, high=num_samps - size + 1)
        end_pos = start_pos + size
        return sample[start_pos:end_pos]

    @staticmethod
    def _random_resamp_cut(sample: np.ndarray, resamp_ratio_low: float, resamp_ratio_high: float,
                           cut_ratio_low: float, cut_ratio_high: float):
        samp_len = len(sample)
        resamped = UCRDataset._random_resample(sample, resamp_ratio_low, resamp_ratio_high)
        cut = UCRDataset._random_cut(resamped, ratio_low=cut_ratio_low, ratio_high=cut_ratio_high)
        if len(cut) > samp_len:
            return UCRDataset._random_cut_size(sample, size=samp_len)
        elif len(cut) < samp_len:
            return UCRDataset._random_pad(cut, match_length=samp_len)
        else:
            return cut

    @staticmethod
    def _random_resample(sample: np.ndarray, ratio_low: float, ratio_high: float):
        assert ratio_low < ratio_high, f'ratio_low: {ratio_low}, ratio_high: {ratio_high}'
        num_samp = len(sample)
        samp_rate = np.random.uniform(low=ratio_low, high=ratio_high)
        num_samp_after = int(num_samp * samp_rate)

        return signal.resample(sample, num=num_samp_after)

    @staticmethod
    def _random_pad(sample: np.ndarray, match_length: int):
        assert len(sample) <= match_length, f'match_length: {match_length}, sample_length: {len(sample)}'

        # determine the size of left and right padding widths
        to_pad = match_length - len(sample)
        rpad_size = np.random.randint(low=0, high=to_pad)
        lpad_size = to_pad - rpad_size

        return np.pad(sample, pad_width=(rpad_size, lpad_size), mode='constant', constant_values=0)

    @staticmethod
    def _add_axis(sample: np.ndarray):
        return sample[np.newaxis, :]

    @staticmethod
    def _to_float32(sample: np.ndarray):
        return sample.astype(dtype=np.float32)

    def _get_row(self, idx: int):
        row = self.data.iloc[idx].to_numpy()
        class_label, timeseries = row[0], row[1:]
        return class_label, timeseries

    @staticmethod
    def _random_cut(sample: np.ndarray, ratio_low: float, ratio_high=1.0):
        assert ratio_low < ratio_high
        assert ratio_low > 0.0
        assert ratio_high <= 1.0

        num_samps = len(sample)
        cut_ratio = np.random.uniform(low=ratio_low, high=ratio_high)
        num_samps_after = int(num_samps * cut_ratio)

        start_pos = np.random.randint(low=0, high=num_samps - num_samps_after + 1)
        end_pos = start_pos + num_samps_after
        return sample[start_pos: end_pos]

    def _select_idx_except(self, except_idx: int):
        while True:
            selected_idx = np.random.randint(low=0, high=len(self))
            selected_class, selected_data = self._get_row(selected_idx)  # TODO: experimental
            if selected_class != except_idx:
                return selected_idx

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = UCRDataset(root_dir='/Users/dansuh/datasets/ucr-archive/UCRArchive_2018/',
                         dataset_name='GunPoint',
                         train=True,
                         sample_length=100)
