from numpy.lib.financial import ipmt
import torch
from torch._C import dtype
import torch.nn as nn

from .networks import CNN, MLP
from .basic_model import BasicModel
from .logic_net_complex import DeepLogic, LogicLayer


class DLN(BasicModel):
    r"""
    A basic deep logic model for simple rule system
    """
    def __init__(self, args):
        super().__init__(args)
        self.name = args.alias
        args = args.model
        self.encoder = CNN(args.encoder)
        self.symbolizers = torch.nn.ModuleDict(
            {
                key: torch.nn.Sequential(
                    nn.Linear(args.encoder.output_size, args.encoder.output_size),
                    nn.ReLU(),
                    nn.Linear(args.encoder.output_size, val.num_output),
                    nn.LogSoftmax(dim=1),
                )
                for key, val in args.symbolizer.items()
            }
        )
        self.inputlayer = LogicLayer(args.deeplogic['INPUT_NUM'], is_leaf=True)
        self.deeplogics = torch.nn.ModuleDict(
            {
                key: DeepLogic(args, name=key, inputlayer=self.inputlayer)
                for key, args in args.deeplogic.items() if key != 'INPUT_NUM'
            }
        )
        # Initial reward mean
        self.reward_mean = {
            "L1": 0.5, "L2": 0.5, "L3": 0.5
        }

    def obtain_optimizer(self):
        return (torch.optim.Adam([{'params': self.encoder.parameters()},
                                  {'params': self.symbolizers.parameters()}
                                  ], lr=1e-4),
                torch.optim.Adam([{'params': self.deeplogics.parameters()},
                                  ], lr=1e-4)
        )
    def vis_connection(self):
        for logics in self.deeplogics.values():
            logics.vis_connection()

    def apply_best_model(self):
        for deeplogic in self.deeplogics.values():
            deeplogic.apply_best_model()

    def forward(self, inputs, aux=None):
        """
        inputs: B * K * C * H * W (image shape)
        """
        B, K, C, H, W = inputs.shape
        hidden = self.encoder(inputs.view(B * K, C, H, W))  # B * K, -1
        symbol_probs = [
            symbolizer(hidden.view(B * K, -1))
            for symbolizer in self.symbolizers.values()
        ]
        symbol_probs = [symbol.reshape(B, K, -1) for symbol in symbol_probs]  # [B, K, N]  prob

        results = [
            deeplogic(symbol_prob, aux)
            for symbol_prob, deeplogic in zip(symbol_probs, self.deeplogics.values())
        ]
        return zip(*results)  # raw logic pred, raw logic accuracy ( if aux given)

    def resample_structure(self):
        loss = 0
        for logicnet in self.deeplogics.values():
            if not logicnet.well_trained:
                logicnet.self_evaluate()
            if logicnet.well_trained:
                logicnet.apply_best_model()
            else:
                loss += logicnet.sample_actions()
        return loss

    def compute_pretrain_loss(self, inputs, meta_data=None, logics=None):
        B, K, C, H, W = inputs.shape
        hidden = self.encoder(inputs.view(B * K, C, H, W))  # B * K, -1
        symbol_probs = {
            key: symbolizer(hidden)
            for key, symbolizer in self.symbolizers.items()
        }
        meta_data = {
            key: val.view(B * K)
            for key, val in meta_data.items()
        }
        # import pdb; pdb.set_trace()
        loss = {
            key: nn.functional.cross_entropy(symbol_probs[key], meta_data[key])
            for key in meta_data
        }

        pred = {key: symbol_probs[key].max(dim=1)[1] for key in meta_data}
        acc = {key: torch.mean((pred[key] == meta_data[key]).float()) for key in meta_data}
        loss_dict = {'loss_{}'.format(key): val.item() for key, val in loss.items()}
        acc_dict = {'acc_{}'.format(key): val.item() for key, val in acc.items()}
        return sum([val for val in loss.values()]), {**loss_dict, **acc_dict}

    def compute_perception_loss(self, inputs, meta_data=None, logics=None):
        """
        compute the losses for the nn model and the logical model
        """
        B, K, C, H, W = inputs.shape

        hidden = self.encoder(inputs.view(B * K, C, H, W))

        symbol_probs = {
            key: symbolizer(hidden).view(B, K, -1)
            for key, symbolizer in self.symbolizers.items()
        }  # the output of encoder

        for idx, key1 in enumerate(self.deeplogics):
            for key2 in symbol_probs:
                logics_idx = torch.eye(3, dtype=torch.bool).cuda()[logics[key2][:, 0]]  # 64 , 3
                logics_mask = logics_idx[:, idx].bool()  # whether this logic
                logics_label = logics_idx[:, idx] * logics[key2][:, 2]  # 64
                self.deeplogics[key1].forward(symbol_probs[key2], sym_id=key2)  # forward
                self.deeplogics[key1].leaf_pruning(logics_label, logics_mask, key2)  # back tracing

        import pdb; pdb.set_trace()
        self.inputlayer[0].pseudo_label['NUMBER']
        for key in symbol_probs:  # sample pseudo_labels
            self.inputlayer

        symbol_acc = {key: self.inputlayer.node[key] == meta_data[key] for key in meta_data}

    def compute_rigger_loss(self, inputs, meta_data=None, logics=None):
        B, K, C, H, W = inputs.shape
        normalize_loss = self.resample_structure()

        self.eval()
        with torch.no_grad():
            hidden = self.encoder(inputs.view(B * K, C, H, W))
            symbol_probs = {
                key: symbolizer(hidden).view(B, K, -1)
                for key, symbolizer in self.symbolizers.items()
            }
            results = {
                key1: {
                    key2: self.deeplogics[key1](symbol_probs[key2], meta_data[key2], key2)
                    for key2 in symbol_probs
                }
                for key1 in self.deeplogics
            }
        self.train()
        # import pdb; pdb.set_trace()
        reinforce_loss = 0
        pseudo_loss = 0
        acc = {}
        for idx, key1 in enumerate(self.deeplogics):
            total_reward = 0
            acc[key1] = {}
            for key2 in symbol_probs:
                logics_pred = results[key1][key2][0][0].node[key2]  # output layer contains only 1 logical node
                logics_idx = torch.eye(3, dtype=torch.bool).cuda()[logics[key2][:, 0]]  # 64 , 3
                logics_mask = logics_idx[:, idx].bool()  # whether this logic
                logics_label = logics_idx[:, idx] * logics[key2][:, 2]  # 64
                
                reward = (logics_pred == logics_label).float()[logics_mask]
                acc[key1][key2] = torch.mean(reward.float()) if logics_mask.any() else 0
                # import pdb; pdb.set_trace()
                total_reward += ((reward.mean()).detach() if logics_mask.any() else 0)
            total_reward /= len(symbol_probs)

            self.reward_mean[key1] = self.reward_mean[key1] * 0.95 + total_reward.mean().detach() * 0.05
            # print(total_reward)
            self.deeplogics[key1].reward_backpropagate(total_reward - self.reward_mean[key1])
            pseudo_loss += self.deeplogics[key1].best_model_tracing(total_reward)
            reinforce_loss += self.deeplogics[key1].rl_loss_gathering()

        lacc_dict = {
            'lacc_{}'.format(key2): sum([acc[key1][key2] for key1 in self.deeplogics]) / 3
            for key2 in symbol_probs
        }
        
        node_pred = {key: torch.stack([x.node[key] for x in self.inputlayer], dim=1) for key in meta_data}
        # import pdb; pdb.set_trace()
        acc_dict = {
            'acc_{}'.format(key): torch.mean((node_pred[key] == meta_data[key]).float())
            for key in meta_data
        }
        loss = 0 * normalize_loss + 10 * reinforce_loss + 1 * pseudo_loss
        return loss, {**lacc_dict, **acc_dict}

    def compute_overall_loss(self, inputs, meta_data=None, logics=None):

        B, K, C, H, W = inputs.shape
        hidden = self.encoder(inputs.view(B * K, C, H, W))

        symbol_probs = {
            key: symbolizer(hidden).view(B, K, -1)
            for key, symbolizer in self.symbolizers.items()
        }  # the output of encoder

        normalize_loss = self.resample_structure()

        logic_acc = {key: {} for key in self.deeplogics}
        total_reward = {key: 0 for key in self.deeplogics}
        for key2 in symbol_probs:
            for idx, key1 in enumerate(self.deeplogics):
                # forward_pass
                logic = self.deeplogics[key1].forward(symbol_probs[key2], sym_id=key2)  # forward

                logics_idx = torch.eye(3, dtype=torch.bool).cuda()[logics[key2][:, 0]]  # 64 , 3
                logics_mask = logics_idx[:, idx].bool()  # whether this logic
                logics_label = logics_idx[:, idx] * logics[key2][:, 2]  # 64
                # well_trained leading to perception learning
                if self.deeplogics[key1].well_trained:
                    self.deeplogics[key1].leaf_pruning(logics_label, logics_mask, key2)
                # compute perception_loss here
                else:
                    logics_pred = logic.node[key2]
                    reward = (logics_pred == logics_label).float()[logics_mask]
                    logic_acc[key1][key2] = torch.mean(reward.float()) if logics_mask.any() else 0
                    total_reward[key1] += ((reward.mean()).detach() if logics_mask.any() else 0)
        
        # Loss computation
        pseudo_loss, reinforce_loss, perception_loss = 0, 0, 0

        for key in self.deeplogics:
            # reward_bp
            if not self.deeplogics[key].well_trained:
                total_reward[key] /= 3
                self.reward_mean[key] = self.reward_mean[key] * 0.95 + total_reward[key] * 0.05
                self.deeplogics[key].reward_backpropagate(total_reward[key] - self.reward_mean[key])
                pseudo_loss += self.deeplogics[key].best_model_tracing(total_reward[key])
                reinforce_loss += self.deeplogics[key].rl_loss_gathering()
        # import pdb; pdb.set_trace()
        for key, probs in enumerate(symbol_probs):
            pseudo_label, mask = self.inputlayer.try_to_find_pseudo_label()
            if mask.any():
                perception_loss += nn.functional.nll_loss(probs[mask], pseudo_label[mask])

        # Accuracy Tracing
        logic_acc_dict = {
            'lacc_{}'.format(key2): sum([logic_acc[key1][key2] for key1 in self.deeplogics]) / 3
            for key2 in symbol_probs
        }

        node_pred = {
            key: torch.stack([x.node[key] for x in self.inputlayer], dim=1)for key in meta_data
        }
        acc_dict = {
            'acc_{}'.format(key): torch.mean((node_pred[key] == meta_data[key]).float())
            for key in meta_data
        }
        reward_dict = {
            'reward_{}'.format(key): val.item() for key, val in self.reward_mean.items()
        } 
       
        logic_loss = 0 * normalize_loss + 10 * reinforce_loss + 1 * pseudo_loss
        return logic_loss, perception_loss, {**logic_acc_dict, **acc_dict, **reward_dict}