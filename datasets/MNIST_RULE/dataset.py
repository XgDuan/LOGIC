import random

import numpy as np
from numpy.core.numeric import indices
import torch
from torch.utils.data import Dataset
import torchvision

import tqdm
from PIL import Image
from .const import COLOR_WHEEL, BCOLOR_WHEEL, RULES2INDEX
from .meta_data_generator import dataset_generation


class MNISTRPM(Dataset):
    def __init__(
        self, root_dir, dataset_size, seq_len, train=True, image_num=6000
    ):
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``EMNIST/processed/test.pt`` exist.
            dataset_size (int), the numbers of data used to train the model
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
        """
        super().__init__()

        self.root_dir = root_dir
        self.dataset_size = dataset_size

        self.mnist_digits = torchvision.datasets.MNIST(
            self.root_dir, train=train, download=True
        )
        self.mnist_inv_tab = {i: [] for i in range(10)}
        for item in self.mnist_digits:
            # We would only use the first dataset_size number of images for training.
            if len(self.mnist_inv_tab[item[1]]) < image_num or not train:
                self.mnist_inv_tab[item[1]].append(item[0])
        self.seq_len = seq_len
        # Equations
        self.datasets = dataset_generation(dataset_size, seq_len)
        random.shuffle(self.datasets)

    def __len__(self):
        return len(self.datasets)

    def render(self, attr_list):
        r"""
            attr_list: [
                [x, x, x]   # attribute 1
                [x, x, x]   # attribute 2
                [x, x, x]   # attribute 3
            ]
            attr_rule: [
                (attr, rule, valid)
            ]
        """
        def number_render(number_mat):
            return [
                np.array(random.choice(self.mnist_inv_tab[i])) for i in number_mat
            ]
        
        def color_render(image_mat, color_mat):
            # return [
            #     image[np.newaxis, :, :] * np.array(COLOR_WHEEL[color])[:, np.newaxis, np.newaxis] + \
            #     (255 - image[np.newaxis, :, :]) * np.array(BCOLOR_WHEEL[bcolor])[:, np.newaxis, np.newaxis]
            #     for image, color, bcolor in zip(image_mat, color_mat, bcolor_mat)
            # ]
            return [
                (image[np.newaxis, :, :] * np.array(COLOR_WHEEL[color])[:, np.newaxis, np.newaxis])
                for image, color in zip(image_mat, color_mat)
            ]
        # import pdb; pdb.set_trace()
        image_mat = color_render(number_render(attr_list[0]), attr_list[1])
        return image_mat

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        # try:
        attr_list, attr_rule = self.datasets[idx]
        # import pdb; pdb.set_trace()
        image_mat = self.render(attr_list)

        # import pdb; pdb.set_trace()
        image_mat = torch.from_numpy(np.stack(image_mat, axis=0)).float() / 255
        # import pdb; pdb.set_trace()
        attr_dict, attr_rule_dict = {}, {}
        # import pdb; pdb.set_trace()
        for al, ar in zip(attr_list, attr_rule):
            attr_dict[ar[0]] = np.array(al, dtype=np.int64)
            # import pdb; pdb.set_trace()
            attr_rule_dict[ar[0]] = np.array([
                RULES2INDEX[ar[1]], ar[2], ar[3]
            ], dtype=np.int64)

        # import pdb; pdb.set_trace()
        return image_mat, attr_dict, attr_rule_dict


if __name__ == '__main__':
    random.seed(1)
    dataset = MNISTRPM('./', 1000, 3)
    import h5py
    F = h5py.File('test_num_rule.hdf5', 'w')
    F.create_group("train")
    F.create_group("attr_mat")
    F["attr_mat"].create_group("NUMBER")
    F["attr_mat"].create_group("COLOR")
    F.create_group("logic_mat")
    F["logic_mat"].create_group("NUMBER")
    F["logic_mat"].create_group("COLOR")
    for idx, data in enumerate(dataset):
        F['train'].create_dataset(str(idx), data=data[0])
        for key, val in data[1].items():
            F['attr_mat'][key].create_dataset(str(idx), data=val)
        for key, val in data[2].items():
            F['logic_mat'][key].create_dataset(str(idx), data=val)
        print(idx)
    F.close()
