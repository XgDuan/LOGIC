import torch
import torch.optim as optim
import torch.nn.functional as F

from .networks import CNN, MLP
from .basic_model import BasicModel


class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args)
        self.conv = CNN(args.input_size, args.hidden_size)
        self.row_feature_aggregate = MLP(
            (args.seq_len * 3 + args.ans_dim - 1) * args.hidden_size, 4 * args.hidden_size
        )
        self.classifizer = MLP(4 * args.hidden_size, args.ans_dim)

        self.optimizer = optim.Adam(
            self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon
        )

    def compute_loss(self, output, target):
        gt_value = output
        zeros = torch.zeros_like(gt_value)
        zeros.scatter_(1, target.view(-1, 1), 1.0)
        return F.binary_cross_entropy_with_logits(gt_value, zeros)
        # loss = F.cross_entropy(output, target)
        # return loss

    def forward(self, input, candidate):
        r"""
        x: input images. B * 3*K * C * H * W
        """
        concat = torch.cat([input, candidate], dim=1)  # B, 3*K + 8, C, H, W
        B, K, C, H, W = concat.shape
        cnn_feature = self.conv(concat.reshape(B * K, C, H, W))  # B, 3*K + 8, H
        cnn_feature = cnn_feature.reshape(B, -1)
        # import pdb; pdb.set_trace()
        fc_feature = self.row_feature_aggregate(cnn_feature)  # B, 4 * hidden
        score = self.classifizer(fc_feature)
        return score
