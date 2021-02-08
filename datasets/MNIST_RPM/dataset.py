import random

import numpy as np
from numpy.core.numeric import indices
import torch
from torch.utils.data import Dataset
import torchvision

import tqdm
from PIL import Image
from .const import COLOR_WHELL, BCOLOR_WHELL
from .meta_data_generator import dataset_generation


class MNIST_RPM(Dataset):
    def __init__(
        self, root_dir, dataset_size, seq_len, train=True, image_num=10000
    ):
        """
        Args:
            root (string): Root directory of dataset where ``EMNIST/processed/training.pt``
                and  ``EMNIST/processed/test.pt`` exist.
            dataset_size (int), the numbers of data used to train the model
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
        """
        super().__init__()

        self.root_dir = root_dir
        self.dataset_size = dataset_size

        # MNIST digits ConstructiondE
        # self.mnist_digits = torchvision.datasets.EMNIST(
        #     self.root_dir,
        #     split='mnist', train=train, download=True
        # )
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

    def __len__(self):
        return self.dataset_size

    def render(self, attr_matrix, wrong_ans):
        # attr_matrix: NUMBER, COLOR, ROTATION. 3  * H  * W
        def number_render(number_mat):
            return [
                np.array(random.choice(self.mnist_inv_tab[i])) for i in number_mat
            ]
        
        def color_render(image_mat, color_mat, bcolor_mat):
            # return [
            #     image[:, :, np.newaxis] * np.array(COLOR_WHELL[color])[np.newaxis, np.newaxis, :] + \
            #     (255 - image[:, :, np.newaxis]) * np.array(BCOLOR_WHELL[bcolor])[np.newaxis, np.newaxis, :]
            #     for image, color, bcolor in zip(image_mat, color_mat, bcolor_mat)
            # ]
            return [
                image[np.newaxis, :, :] * np.array(COLOR_WHELL[color])[:, np.newaxis, np.newaxis] + \
                (255 - image[np.newaxis, :, :]) * np.array(BCOLOR_WHELL[bcolor])[:, np.newaxis, np.newaxis]
                for image, color, bcolor in zip(image_mat, color_mat, bcolor_mat)
            ]
            # return [
            #     image[:, :, np.newaxis] * np.array(COLOR_WHELL[color])[np.newaxis, np.newaxis, :]
            #     for image, color in zip(image_mat, color_mat)
            # ]
        # def bcolor_render(image_mat, color_mat):
        #     res = []
        #     for image, color in zip(image_mat, color_mat):
        #         new_image = image + image * np.array(BCOLOR_WHELL[color])[np.newaxis, np.newaxis, :]
        #     return [
        #         image * 
        #         for 
        #     ]    
    
        def rotation_render(image_mat, rotation_mat):
            return [
                np.rot90(image, rotation, axis=(1, 2))
                for image, rotation in zip(image_mat, rotation_mat)
            ]
        # print(attr_matrix)
        image_mat = number_render(attr_matrix[0])
        image_mat = color_render(image_mat, attr_matrix[1], attr_matrix[2])
        # import pdb; pdb.set_trace()
        # image_mat = rotation_render(image_mat, attr_matrix[2])
        wrong_ans_img = number_render(wrong_ans[0])
        wrong_ans_img = color_render(wrong_ans_img, wrong_ans[1], attr_matrix[2])
        # wrong_ans_img = rotation_render(wrong_ans_img, wrong_ans[2])

        return image_mat, wrong_ans_img

    def __getitem__(self, idx):
        attr_mat, wrong_ans = self.datasets[idx]

        image_mat, wrong_ans = self.render(attr_mat, wrong_ans)

        # move the answer from image_mat to answer_mat
        answer_mat = [image_mat[-1]] + wrong_ans
        image_mat = image_mat[:-1]

        # resize
        resize_image = []
        resize_ans = []
        for idx in range(self.seq_len * 3 - 1):
            resize_image.append(
                np.array(Image.fromarray(np.uint8(image_mat[idx]), 'RGB').resize((80, 80)))
            )
        image_mat = torch.from_numpy(np.transpose(np.float32(np.stack(resize_image)), (0, 3, 1, 2)))

        for idx in range(8):
            resize_ans.append(
                np.array(Image.fromarray(np.uint8(answer_mat[idx]), 'RGB').resize((80, 80)))
            )
        answer_mat = torch.from_numpy(np.transpose(np.float32(np.stack(resize_ans)), (0, 3, 1, 2)))

        # n_images * h * w * c
        # image_mat = torch.from_numpy(np.stack(image_mat, axis=0))
        # answer_mat = torch.from_numpy(np.stack(answer_mat, axis=0))

        indices = list(range(answer_mat.shape[0]))
        np.random.shuffle(indices)
        target = 0  # The answer is always the first one in answer_mat
        new_target = indices.index(target)
        new_answer_mat = answer_mat[indices, :, :, :]

        # return two matrix and the rules and an answer (As pytorch tensors)
        return image_mat, new_answer_mat, new_target, attr_mat


if __name__ == '__main__':
    dataset = MNIST_RPM('./', 1000)
    import h5py
    F = h5py.File('test.hdf5', 'w')
    F.create_group("train")
    F.create_group("answer")
    for idx, data in enumerate(dataset):
        F['train'].create_dataset(str(idx), data=data[0])
        F['answer'].create_dataset(str(idx), data=data[1])
        print(idx)
    F.close()
    # import pdb; pdb.set_trace()
