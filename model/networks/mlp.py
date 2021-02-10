r"""
A simple CNN model to extract MNIST image features
"""

import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, name='mlp'):
        super().__init__()

        # The classification structure: two layer FC
        self.row_feature = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size * 2, input_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size, output_size)
        )
        self.repr_str = "{}(name={}; {} -> {})".format(
            self.__class__.__name__, name, input_size, output_size)
    
    def __repr__(self):
        return self.repr_str

    def forward(self, inputs):
        r"""
        Args:
            inputs (tensor of size: b * k * 28 * 28):
                the input image stacks, where k is number of NMIST images, b the batchsize
        """
        feature = self.row_feature(inputs)  # B * K, 128
        return feature  # B, K, output_size
        # return self.classifier(feature)
