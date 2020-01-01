from typing import Union, Tuple

from .model import TSNet
from .loss import TripletLoss

import torch
from torch import nn
from torch import optim
from torchland.trainer.trainer import NetworkTrainer, TrainStage

epoch = 2000
lr_init = 3e-4


class TSNetTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()  # TODO:

    def run_step(self,
                 models,
                 criteria,
                 optimizers,
                 input_,
                 train_stage: TrainStage,
                 *args, **kwargs):
        pass
