from .model import TSNet
from .loss import TripletLoss
from .dataset import UCRDataLoaderBuilder

import torch
from torch import optim
from torchland.trainer.trainer import NetworkTrainer, TrainStage
import numpy as np


class TSNetTrainer(NetworkTrainer):
    def __init__(self, config: dict):
        super().__init__(epoch=2000)  # TODO:
        self.lr_init = self.get_or_else(
            config, 'lr_init', default=3e-4)
        self.batch_size = self.get_or_else(
            config, 'batch_size', default=10)
        data_root = self.get_or_else(
            config, 'data_root', default='data_in')
        data_name = self.get_or_else(
            config, 'data_name', default=None)  # must be specified
        self.sample_length = 150

        self.input_size = (1, self.sample_length)

        model = TSNet(in_channel=1, middle_channel=20, out_channel=20, num_layers=10)
        self.add_model('tsnet', model, input_size=self.input_size)
        self.add_optimizer('adam', optim.Adam(model.parameters(), lr=self.lr_init))
        # self.add_criterion('triplet_loss', TripletLoss())  # TODO: redefine loss

        self.dataloader_maker = UCRDataLoaderBuilder(data_root, data_name, self.batch_size)
        self.set_dataloaders(self.dataloader_maker)

    @staticmethod
    def get_or_else(config: dict, key, default):
        return config[key] if key in config else default

    def num_samples(self, train_stage: TrainStage):
        if train_stage == TrainStage.TRAIN:
            return self.dataloader_maker.num_train
        elif train_stage == TrainStage.VALIDATE:
            return self.dataloader_maker.num_val
        elif train_stage == TrainStage.TEST:
            return self.dataloader_maker.num_test
        raise ValueError(f'Invalid train_stage value: {train_stage}')

    def run_step(self,
                 models,
                 criteria,
                 optimizers,
                 input_,
                 train_stage: TrainStage,
                 *args, **kwargs):
        class_labels, in_data = input_
        print(input_)

        out = models.tsnet.model(in_data)

        # calculate loss
        loss = torch.randn(1, 2, 3)  # TODO: actually calculate this

        num_data_samples = self.num_samples(train_stage)

        compared_length = None  # ???
        num_random_samples = 12
        sample_data = np.random.choice(
            num_data_samples, size=(num_random_samples, self.batch_size))

        example_length = np.random.randint(low=1, high=self.sample_length + 1)

        anchor_length = np.random.randint(
            low=example_length, high=self.sample_length + 1)
        anchor_start_pos = np.random.randint(
            low=0, high=self.sample_length - example_length + 1, size=self.batch_size)

        pos_start_offset = np.random.randint(
            low=0, high=self.sample_length - example_length + 1, size=self.batch_size)
        pos_start_pos = anchor_start_pos + pos_start_offset

        # calculate anchor, positive, negative representations

        if train_stage == TrainStage.TRAIN:
            adam = optimizers.adam
            adam.zero_grad()
            loss.backward()
            adam.step()


if __name__ == '__main__':
    trainer = TSNetTrainer(config={})
    trainer.fit()
