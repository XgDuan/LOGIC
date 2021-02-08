import torch
import torch.optim as optim
import torch.nn.functional as F

from .networks import CNN, MLP
from .basic_model import BasicModel


class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args)
        self.conv = CNN(args.input_size, args.hidden_size)
        self.row_feature_aggregate = MLP(3 * args.seq_len * args.hidden_size, args.hidden_size)
        self.classifizer = MLP(args.hidden_size, 1)

        self.optimizer = optim.Adam(
            self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon
        )

    def compute_loss(self, output, target):
        # pred = output
        # gt_value = output
        # zeros = torch.zeros_like(gt_value)
        # zeros.scatter_(1, target.view(-1, 1), 1.0)
        # return F.binary_cross_entropy_with_logits(gt_value, zeros)

        loss = F.cross_entropy(output, target)
        return loss

    def forward(self, input, candidate):
        r"""
        x: input images. B * 3*K * C * H * W
        """
        _, Ki, C, H, W = input.shape
        concat = torch.cat((input, candidate), dim=1)  # B, 3 * K + 7, C, H, W
        B, Ka, C, H, W = concat.shape
        K = Ka - Ki
        cnn_feature = self.conv(concat.reshape(B * Ka, C, H, W))  # B, 3*K + 7, W, H
        cnn_feature = cnn_feature.reshape(B, Ka, -1)
        reference, candidate = cnn_feature.split(dim=1, split_size=(Ki, K))
        reference = torch.cat([
            reference.unsqueeze(1).expand(-1, K, -1, -1),
            candidate.unsqueeze(2).expand(-1, -1, -1, -1)
        ], dim=2).reshape(B * K, -1)
        new_hidden = self.row_feature_aggregate(reference)
        # import pdb; pdb.set_trace()
        score = self.classifizer(new_hidden)
        score = score.reshape(B, K)
        # import pdb; pdb.set_trace()
        return score
