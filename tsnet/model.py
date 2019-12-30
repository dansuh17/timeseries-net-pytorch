from typing import List
import torch
from torch import nn


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        # calculate the padding size to match the size of the input length and output length
        # - makes it causal
        padding = (kernel_size - 1) * dilation
        pad_dims = (padding, 0)  # only on the left side

        # dilated causal network
        self.net = nn.Sequential(
            # sub-block 1
            nn.ConstantPad1d(pad_dims, value=0),
            nn.utils.weight_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # sub-block 2
            nn.ConstantPad1d(pad_dims, value=0),
            nn.utils.weight_norm(
                nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # resampling with kernel size 1 is required if input channels and output channels differ
        self.resample: bool = in_channels != out_channels
        self.conv_resample = None
        if self.resample:
            self.conv_resample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if self.resample:
            residual = self.conv_resample(x)
        else:
            residual = x
        return self.net(x) + residual


class SqueezeTimeChannel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze(dim=2)


class CausalCNN(nn.Module):
    def __init__(self, in_channel, out_channels: List[int], dilations: List[int]):
        super().__init__()

        assert len(out_channels) == len(dilations), \
            f'Number of elements in out_channels and dilations must match: {len(out_channels)} vs {len(dilations)}'

        dilation_blocks = []
        for out_channel, dilation in zip(out_channels, dilations):
            dilation_blocks.append(
                CausalConvBlock(in_channel, out_channel, kernel_size=3, dilation=dilation))
            in_channel = out_channel  # update input channel number
        self.net = nn.Sequential(*dilation_blocks)

    def forward(self, x):
        return self.net(x)


def doubled_dilations(num_layers: int):
    return [2 ** i for i in range(num_layers)]


class TSNet(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel, num_layers: int):
        super().__init__()

        # equal numbers of output channels
        middle_channels = [middle_channel] * num_layers

        self.net = nn.Sequential(
            CausalCNN(in_channel, middle_channels, dilations=doubled_dilations(num_layers)),
            nn.AdaptiveAvgPool1d(output_size=1),  # reduce size
            SqueezeTimeChannel(),  # reduce dimensions
            nn.Linear(in_features=middle_channel, out_features=out_channel),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    net = TSNet(in_channel=1, middle_channel=2, out_channel=123, num_layers=10)

    dummy = torch.randn((1, 1, 2000))
    out = net(dummy)

    print(out.size())
