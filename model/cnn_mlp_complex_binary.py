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
        self.logic_classifier = torch.nn.ModuleDict({
            '%s-%s'.format(key1, key2): MLP(args.encoder.output_size * 3, 2)  # three
            for key1 in ['NUMBER', 'COLOR'] for key2 in ['L1', 'L2', 'L3']
        })

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
            loss_logic = 0
            logic_acc = 0
            for key1 in ['NUMBER', 'COLOR']:
                for idx, key2 in enumerate(['L1', 'L2','L3']):
                    pred = logic_pred['%s-%s'.format(key1, key2)]
                    # import pdb; pdb.set_trace()
                    logics_idx = torch.eye(3, dtype=torch.bool).cuda()[logics[key1][:, 0]]
                    logics_mask = logics_idx[:, idx].bool()  # whether this logic
                    logics_label = logics_idx[:, idx] * logics[key1][:, 2]  # 64
                    loss_logic += F.cross_entropy(pred[logics_mask], logics_label[logics_mask])
                    logic_acc += (pred[logics_mask].max(dim=-1)[1] == logics_label[logics_mask]).float().mean()
            loss += loss_logic
            lacc_dict['lacc_NUMBER'] = logic_acc / 6
        
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
        logics = {
            key: val(hidden.view(B, -1)) for key, val in self.logic_classifier.items()
        }
        return number, color, logics