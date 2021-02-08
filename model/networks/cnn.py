r"""
A simple CNN model to extract MNIST image features
"""

import torch
import torch.nn as nn


class CNN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.input_size == [28, 28]  # the input size should be 28 as a constant
        # The CNN Structure from pytorch tutorial:
        # https://github.com/pytorch/examples/blob/master/mnist/main.py

        self.conv = nn.Sequential(
            nn.Conv2d(args.input_channel, 64, 3, 1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216 * 2, args.output_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )  # 28 * 28 * 1 -> 128

        # The classification structure: two layer FC
        # self.classifier = nn.Sequential(
        #     nn.Linear(3 * 128, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(128, 1)
        # )

    def forward(self, inputs):
        r"""
        Args:
            inputs (tensor of size: b * C * 28 * 28):
                the input image stacks
        """
        B, C, H, W = inputs.size()
        feature = self.conv(inputs)  # B * K, 128
        return feature  # BK, 128
        # return self.classifier(feature)


class DCNN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.output_size == [28, 28]

        self.dconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 32, 3, 2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 8, 3, 2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, args.output_channel, 3, 1),
        )
    
    def forward(self, inputs):
        r"""
        """
        B, H = inputs.size()
        output = self.dconv(inputs.unsqueeze(2).unsqueeze(3))
        return output
