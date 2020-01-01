import pandas as pd
import os
from torch.utils.data import dataset


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

    def __getitem__(self, idx):
        return self.data.iloc[idx].to_numpy()

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = UCRDataset(root_dir='/Users/dansuh/datasets/ucr-archive/UCRArchive_2018/',
                         dataset_name='GunPoint',
                         train=True)
