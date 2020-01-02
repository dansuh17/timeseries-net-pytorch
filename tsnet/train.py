from .model import TSNet
from .dataset import UCRDataLoaderBuilder

import torch
from torch import optim, nn
from torchland.trainer.trainer import NetworkTrainer, TrainStage


class TSNetTrainer(NetworkTrainer):
    def __init__(self, config: dict):
        super().__init__(epoch=2000)
        self.lr_init = config.get('lr_init', 3e-4)
        self.batch_size = config.get('batch_size', 10)
        data_root = config.get('data_root', 'data_in')
        data_name = config.get('data_name', None)  # must be specified
        self.sample_length = 100

        self.input_size = (1, self.sample_length)

        model = TSNet(in_channel=1, middle_channel=20, out_channel=20, num_layers=10)
        self.add_model('tsnet', model, input_size=self.input_size)
        self.add_optimizer('adam', optim.Adam(model.parameters(), lr=self.lr_init))
        # self.add_criterion('triplet_loss', TripletLoss())  # TODO: redefine loss

        self.dataloader_maker = UCRDataLoaderBuilder(
            data_root, data_name, self.batch_size, sample_length=self.sample_length)
        self.set_dataloaders(self.dataloader_maker)

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
        class_labels, anchors, positive_samples, negative_samples = input_

        encoder = models.tsnet.model

        z_anchors = encoder(anchors)
        z_positive = encoder(positive_samples)
        z_negative = encoder(negative_samples)

        # calculate loss
        dotprod_pos = torch.bmm(z_anchors.view(self.batch_size, 1, -1),
                                z_positive.view(self.batch_size, -1, 1))
        dotprod_neg = torch.bmm(z_anchors.view(self.batch_size, 1, -1),
                                z_negative.view(self.batch_size, -1, 1))

        positive_loss = -torch.mean(nn.functional.logsigmoid(dotprod_pos))
        negative_loss = torch.mean(nn.functional.logsigmoid(dotprod_neg))
        total_loss = positive_loss + negative_loss

        if train_stage == TrainStage.TRAIN:
            adam = optimizers.adam
            adam.zero_grad()
            total_loss.backward()
            adam.step()

        return (z_anchors, z_positive, z_negative), total_loss


if __name__ == '__main__':
    test_config = {
        'data_name': 'GunPoint',
        'data_root': '/Users/dansuh/datasets/ucr-archive/UCRArchive_2018/',
    }
    trainer = TSNetTrainer(config=test_config)
    trainer.fit()
