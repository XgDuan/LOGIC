import torch
import torch.optim as optim
import torch.nn.functional as F

from .networks import CNN, MLP
from .basic_model import BasicModel


class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args)
        self.conv = CNN(args.input_size, args.hidden_size)
        self.class_number = MLP(args.hidden_size, 10)  # num
        self.class_forcol = MLP(args.hidden_size, 12)  # forecolor
        self.class_baccol = MLP(args.hidden_size, 11)  # backcolor

        self.optimizer = optim.Adam(
            self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon
        )

    def compute_loss(self, output, attr_mat):
        loss_number = F.cross_entropy(output[0], attr_mat[:, 0, :-1].flatten())
        loss_forcol = F.cross_entropy(output[1], attr_mat[:, 1, :-1].flatten())
        loss_baccol = F.cross_entropy(output[2], attr_mat[:, 2, :-1].flatten())
        loss = loss_number + loss_forcol + loss_baccol
        print("Loss: {}, {}, {}".format(loss_number, loss_forcol, loss_baccol))
        return loss

    def train_(self, image, candidate, target=None, attr_mat=None):
        self.optimizer.zero_grad()
        output = self(image, candidate)
        loss = self.compute_loss(output, attr_mat)
        loss.backward()
        self.optimizer.step()
        pred_number = output[0].data.max(1)[1]
        pred_forcol = output[1].data.max(1)[1]
        pred_baccol = output[2].data.max(1)[1]
        correct_number = pred_number.eq(attr_mat[:, 0, :-1].flatten().data).cpu().sum().numpy()
        correct_forcol = pred_forcol.eq(attr_mat[:, 1, :-1].flatten().data).cpu().sum().numpy()
        correct_bacclo = pred_baccol.eq(attr_mat[:, 2, :-1].flatten().data).cpu().sum().numpy()
        acc_number = correct_number * 100.0 / attr_mat.shape[0] / 8
        acc_forclo = correct_forcol * 100.0 / attr_mat.shape[0] / 8
        acc_bacclo = correct_bacclo * 100.0 / attr_mat.shape[0] / 8
        
        return loss.item(), acc_number, acc_forclo, acc_bacclo

    def validate_(self, image, candidate, target=None, attr_mat=None):
        with torch.no_grad():
            output = self(image, candidate)
        loss = self.compute_loss(output, attr_mat)
        pred_number = output[0].data.max(1)[1]
        pred_forcol = output[1].data.max(1)[1]
        pred_baccol = output[2].data.max(1)[1]
        correct_number = pred_number.eq(attr_mat[:, 0, :-1].flatten().data).cpu().sum().numpy()
        correct_forcol = pred_forcol.eq(attr_mat[:, 1, :-1].flatten().data).cpu().sum().numpy()
        correct_bacclo = pred_baccol.eq(attr_mat[:, 2, :-1].flatten().data).cpu().sum().numpy()
        acc_number = correct_number * 100.0 / attr_mat.shape[0] / 8
        acc_forclo = correct_forcol * 100.0 / attr_mat.shape[0] / 8
        acc_bacclo = correct_bacclo * 100.0 / attr_mat.shape[0] / 8
        
        return loss.item(), acc_number, acc_forclo, acc_bacclo

    def forward(self, input, candidate):
        r"""
        x: input images. B * 3*K * C * H * W
        """
        B, K, C, H, W = input.shape
        cnn_feature = self.conv(input.reshape(B * K, C, H, W))  # => b*K, h
        number = self.class_number(cnn_feature)  # B * K, -1
        forcol = self.class_forcol(cnn_feature)  # B * K, -1
        baccol = self.class_baccol(cnn_feature)  # B * K, -1

        return [number, forcol, baccol]
