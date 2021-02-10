import torch
import torch.optim as optim
import torch.nn.functional as F

from .networks import CNN, MLP
from .basic_model import BasicModel


class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args)
        args = args.model
        # import pdb; pdb.set_trace()
        self.conv = CNN(args.encoder)
        # self.features = MLP(args.encoder.output_size, args.encoder.output_size)
        self.number_classifier = MLP(args.encoder.output_size, 10)  # two attributes
        self.color_classifier = MLP(args.encoder.output_size, 10)
        self.logic_classifier = MLP(args.encoder.output_size * 3, 6)  # three rules for two attributes

    def resample_structure(self):
        pass

    def apply_best_model(self):
        pass

    # def obtain_optimizer(self):
    #     return (torch.optim.Adam([{'params': self.conv.parameters()},
    #                               {'params': self.number_classifier.parameters()},
    #                               {'params': self.color_classifier.parameters()},
    #                               {'params': self.logic_classifier.parameters()},
    #                               ]),
    #             torch.optim.Adam([{'params': self.color_classifier.parameters()},
    #                               ])
    #     )

    def compute_overall_loss(self, inputs, meta_data=None, logics=None):
        r"""
        inputs: B, K, C, H, W
        meta_data: 
        """
        B, K, _, _, _ = inputs.shape
        number, color, logic_pred = self.forward(inputs)
        # import pdb; pdb.set_trace()

        acc_dict = {}
        lacc_dict = {}

        loss = 0
        if meta_data is not None:
            # meta_target = torch.stack([
            #     meta_data['NUMBER'], meta_data['COLOR']
            # ], dim=2).view(B * K * 2)  # B * K * 2
            # attr_pred = attr_pred.view(B * K * 2, -1)
            loss_num = F.cross_entropy(number.view(B * K, 10), meta_data['NUMBER'].view(B * K))
            loss_col = F.cross_entropy(color.view(B * K, 10), meta_data['COLOR'].view(B * K))
            # loss += loss_num + loss_col
            acc_dict['acc_NUMBER'] = (number.max(dim=-1)[1] == meta_data['NUMBER']).float().mean()
            acc_dict['acc_COLOR'] = (color.max(dim=-1)[1] == meta_data['COLOR']).float().mean()
        
        # loss_phi = 0
        if logics is not None:
            # import pdb; pdb.set_trace()

            logic_target = torch.stack([
                logics['NUMBER'][:, 2], logics['COLOR'][:, 2]
            ], dim=1).view(B * 2)
            # import pdb; pdb.set_trace()
            logic_pred = logic_pred.view(B * 2, 3)
            loss += F.cross_entropy(logic_pred, logic_target)
            lacc_dict['lacc_NUMBER'] = (logic_pred.max(dim=-1)[1] == logic_target).float().mean()
        
        return loss, 0, {**acc_dict, **lacc_dict}

    
    def forward(self, inputs):
        r"""
        input: B * K * C * H * W
        """
        B, K, C, H, W = inputs.shape
        hidden = self.conv(inputs.view(B * K, C, H, W))
        # hidden = self.features(hidden)
        number = self.number_classifier(hidden).view(B, K, 10)
        color = self.color_classifier(hidden).view(B, K, 10)
        logics = self.logic_classifier(hidden.view(B,  -1)).view(B, 6)
        return number, color, logics