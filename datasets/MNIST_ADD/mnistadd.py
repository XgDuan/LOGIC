r"""
The dataset definition
"""

import random
import logging

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

logger = logging.getLogger('dataset')


def train_test_splits(split_train_test=True):
    """
    Args:
        split_train_test: whether to split the training and testing
    """
    full = {
        1: [[1, 0], [0, 1]],
        2: [[2, 0], [1, 1], [0, 2]],
        3: [[3, 0], [2, 1], [1, 2], [0, 3]],
        4: [[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]],
        5: [[5, 0], [4, 1], [3, 2], [2, 3], [1, 4], [0, 5]],
        6: [[6, 0], [5, 1], [4, 2], [3, 3], [2, 4], [1, 5], [0, 6]],
        7: [[7, 0], [6, 1], [5, 2], [4, 3], [3, 4], [2, 5], [1, 6], [0, 7]],
        8: [[8, 0], [7, 1], [6, 2], [5, 3], [4, 4], [3, 5], [2, 6], [1, 7], [0, 8]],
        9: [[9, 0], [8, 1], [7, 2], [6, 3], [5, 4], [4, 5], [3, 6], [2, 7], [1, 8], [0, 9]]
    }
    train_split, test_split = [], []
    if split_train_test:
        for key, val in full.items():
            for equation in val:
                if random.random() > 0.2:
                    train_split.append([*equation, key])
                else:
                    test_split.append([*equation, key])
    else:
        for key, val in full.items():
            for equation in val:
                train_split.append([*equation, key])
                test_split.append([*equation, key])
    return train_split, test_split


class MNISTAdd(Dataset):
    def __init__(
        self, root_dir, dataset_size, equation_split, negative_type, train=True, image_num=2000
    ):
        """
        Args:
            root (string): Root directory of dataset where ``EMNIST/processed/training.pt``
                and  ``EMNIST/processed/test.pt`` exist.
            dataset_size (int), the numbers of data used to train the model
            equation_split (list): the euqations used to train the model
            negative_type (enumuate): the type of negative samples:
                "+1": wrong answer could be the true answer +1
                "-1": similar
                "rd": random number
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
        """
        super().__init__()

        self.root_dir = root_dir
        self.dataset_size = dataset_size
        self.equation_split = self.add_negative(equation_split, negative_type)
        random.shuffle(self.equation_split)
        self.mnist_digits = torchvision.datasets.MNIST(
            self.root_dir, train=train, download=True
        )
        self.mnist_inv_tab = {i: [] for i in range(10)}
        for item in self.mnist_digits:
            # We would only use the first dataset_size number of images for training.
            if len(self.mnist_inv_tab[item[1]]) < image_num or not train:
                self.mnist_inv_tab[item[1]].append(item[0])^%+

    def __len__(self):
        return self.dataset_size

    def add_negative(self, equation_split, negative_type):
        negative_generator = None
        if negative_type == '+1':
            negative_generator = lambda x: (x + 1) % 10
        elif negative_type == '-1':
            negative_generator = lambda x: (x - 1) % 10
        elif negative_type == 'rd':
            negative_generator = lambda x: (x + random.randint(1, 9)) % 10

        negative_split = []

        for idx in range(len(equation_split)):
            equation = equation_split[idx]
            negative = [equation[0], equation[1], negative_generator(equation[2])]
            equation_split[idx] = (equation, 1)
            negative_split.append((negative, 0))
        return equation_split + negative_split

    def __getitem__(self, idx):
        idx = idx % len(self.equation_split)
        digits = [
            np.array(random.choice(self.mnist_inv_tab[i])) for i in self.equation_split[idx][0]
        ]
        image = torch.tensor(np.stack(digits), dtype=torch.float)
        image = image.unsqueeze(1)  # K * 1 * H * W
        target = torch.tensor(self.equation_split[idx][1], dtype=torch.float)

        return image, target, [self.equation_split[idx][0]]


if __name__ == "__main__":
    train_split, test_split = train_test_splits()
    dataset = MNISTAdd(
        '../data', 100, equation_split=train_split, negative_type='rd', train=False
    )
