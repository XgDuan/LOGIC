import torch
import torch.optim as optim
import torch.nn.functional as F

from .networks import CNN, MLP
from .basic_model import BasicModel


class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args)
        self.conv = CNN(args.input_size, args.hidden_size)
        self.attribute_classifier = MLP(args.hidden_size, 20)  # two attributes
        self.classifier = MLP(args.hidden_size * 3, 12)  # three rules for two attributes


    def obtain_optimizer(self):
        return (torch.optim.Adam([{'params': self.conv.parameters()}]),
                torch.optim.Adam([{'params': self.classifier.parameters()}])
        )

    def compute_loss(self, output, target):
        # pred = output
        # gt_value = output
        # zeros = torch.zeros_like(gt_value)
        # zeros.scatter_(1, target.view(-1, 1), 1.0)
        # return F.binary_cross_entropy_with_logits(gt_value, zeros)

        loss = F.cross_entropy(output, target)
        return loss

    def compute_overall_loss(self, inputs, meta_data=None, logics=None):
        pass
    
    def forward(self, inputs):
        r"""
        input: B * K * C * H * W
        """
        B, K, C, H, W = inputs.shape
        hidden = self.conv(inputs.view(B * K, C, H, W))