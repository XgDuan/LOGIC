import asyncio
import copy
import heapq
import queue
import random
import math
from queue import Queue
from functools import reduce
from queue import PriorityQueue
from typing import Iterator
from itertools import chain

import numpy as np
import torch
from torch._C import dtype
# from torch._C import device, dtype
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from typing_extensions import final
# from torch.autograd import Function
# from torch.nn.functional import grid_sample, layer_norm, relu
from .networks import CNN, MLP, VAE_loss, SymbolNet
from .networks.logic_op import TERM_OP, RELATION_OP, RELATION_INIT_OP, LOGIC_OP, UNITARY_OP

from .basic_model import BasicModel

r"""
In this function, we define the basic

"""

import logging
logger = logging.getLogger(__name__)

HIDDEN_DIM = 256


class LeafNode(nn.Module):
    r"""
    Leafnodes are the connectors between neural networks and logic reasoning systems.
    There are several things to do within them:
        1) sample symbols from neural network outputs.
        2) maintain the samples and their log probs.
    """
    def __init__(self, input_idx):
        super().__init__()
        self.inputs = {}
        self.node = {}
        self.input_idx = input_idx
        self.logic_type = 'LEAF'
        self.log_prob_ = {}

        self.label = {}
        self.pseudo_label = {}
        self.fixed_value = {}

    def log_prob(self):
        return self.log_prob_

    def __repr__(self):
        return "{}(logic_type={})".format(self.__class__.__name__, 'leaf')

    def vis_connection(self, space_num):
        return "[{}]".format(self.input_idx)
        # return "  " * space_num + "({})".format(self.input_idx)

    def forward(self, inputs):
        r"""
        inputs: Batch, num_inputs, num_symbol (log probs of inputs symbols)
        """
        symbol, sym_id = inputs
        self.inputs[sym_id] = symbol[:, self.input_idx, :]  # cached for further revisal

        # TODO: modify this temp solution
        self.log_prob_[sym_id], self.node[sym_id] = self.inputs[sym_id].max(1)
        # if self.training:
        #     m = distributions.Categorical(probs=torch.exp(self.inputs[sym_id]))
        #     self.node[sym_id] = m.sample()
        #     self.log_prob_[sym_id] = m.log_prob(self.node[sym_id])  # used for loss computation and backward
        # else:
        #     # import pdb; pdb.set_trace()
        #     self.log_prob_[sym_id], self.node[sym_id] = self.inputs[sym_id].max(1)
        return [self, sym_id]  # sym_id is used for tracing multiple variable at the same time


class LogicNode(nn.Module):
    r"""
    Logicnodes are the core components of the logic reasoning system.
    They maintain the input and operation probs, calculate the output given inputs and connections.
    Basically, the inputs to logic nodes should be a batch of symbols.

    """
    def __init__(self, last_layer, operation_pool, type,):
        super().__init__()

        self.logic_type = type

        # Input choices TODO: Initialization
        self.last_layer = last_layer  # pointer to previous layer
        self.logits_node_l = nn.parameter.Parameter(torch.rand(HIDDEN_DIM))
        self.logits_node_r = nn.parameter.Parameter(torch.rand(HIDDEN_DIM))

        # logic operation choice
        self.op_pool = operation_pool
        self.logits_op = nn.parameter.Parameter(torch.rand(HIDDEN_DIM))
        self._log_probs = {}  # containers to contain necessary probs
        self.node_idx_l, self.node_idx_r, self.op_idx = None, None, None
        self.log_prob_need_renew = False
        self.node = {}

    def update_log_probs(self, key, val):
        self._log_probs[key] = val
        self.log_prob_need_renew = True

    def log_prob(self, key=None):
        r"""
        """
        if key:
            if key in self._log_probs:
                return self._log_probs[key]
            else:
                assert False, "Key not exists"
        elif self.log_prob_need_renew:
            self._log_prob_ = sum([v for v in self._log_probs.values()])
            self.log_prob_need_renew = False
        return self._log_prob_

    def __repr__(self):
        return "{}(num_input={}, logic_type={}, logic_op=[{}])".format(
            self.__class__.__name__, len(self.last_layer), self.logic_type,
            ", ".join([x.__class__.__name__ for x in self.op_pool]))

    def vis_connection(self, space_num):
        return "{}({}, {})".format(
            self.op_pool[self.op_idx].__class__.__name__,
            self.last_layer[self.node_idx_l].vis_connection(space_num + 1),
            self.last_layer[self.node_idx_r].vis_connection(space_num + 1)
        )

    def forward(self, last_layer):
        sym_id = last_layer[0][1]  # sym_id
        self.node[sym_id] = self.op_pool[self.op_idx](
            last_layer[self.node_idx_l][0].node[sym_id],
            last_layer[self.node_idx_r][0].node[sym_id]
        )
        return [self, sym_id]


class RiggerLayer(nn.Module):
    r"""
    Rigger layers are used to sample logical path.
    """
    def __init__(self, num_input, num_cell, num_op, op_pool, embedding=None, layer_cell=None):
        super().__init__()
        self.num_cell = num_cell
        self.op_pool = op_pool
        self.num_action = num_input * num_input * num_op

        self.embedding = embedding if embedding else nn.Embedding(self.num_action + 1, HIDDEN_DIM)
        self.cell = nn.LSTMCell(HIDDEN_DIM, HIDDEN_DIM)
        self.layer_cell = layer_cell if layer_cell else nn.LSTMCell(HIDDEN_DIM, HIDDEN_DIM)
        self.decoder = nn.Linear(HIDDEN_DIM, self.num_action)
        self.anneal = 1
        self.global_step = 0

    def sample(self, prev_state):
        last_token = torch.LongTensor([self.num_action],).cuda()  # init token
        hidden_0 = prev_state
        hidden_1 = (torch.zeros_like(hidden_0[0]), torch.zeros_like(hidden_0[1]))

        actions = []
        action_probs = []
        raw_probs = []
        layer_loss = 0
        last_probs = None
        
        # anneal
        self.global_step += 1
        self.anneal = math.exp(self.global_step * 0.0001)  # lower
        for i in range(self.num_cell):  # decoding all the actions
            inputs = self.embedding(last_token)
            hidden_0 = self.layer_cell(inputs, hidden_0)
            inputs_1 = hidden_0[0]  # inputs should be the hidden state
            hidden_1 = self.cell(inputs_1, hidden_1)

            output = hidden_1[0]  # output should be the hidden state of the second layer
            raw_prob = self.decoder(output)
            # import pdb; pdb.set_trace()
            raw_prob = self.anneal * torch.tanh(raw_prob)   # flatten
            probs = torch.softmax(raw_prob, dim=1)
            # sampling procedure
            m = Categorical(probs=probs)
            pred = m.sample()
            prob = m.log_prob(pred)  # softmax prob

            actions.append(pred)
            action_probs.append(prob)
            raw_probs.append(raw_prob)
            last_token = pred
            if last_probs is not None:
                layer_loss -= torch.sum(last_probs * torch.log(last_probs)) - torch.sum(last_probs * torch.log(probs))
            last_probs = probs.detach()  # whether to detach or not here?
        return actions, action_probs, raw_probs, hidden_0, layer_loss  # Try this

    def supervised_loss(self, prev_state, sup_seq):
        last_token = torch.LongTensor([self.num_action],).cuda()  # init token
        sup_seq = [last_token] + sup_seq
        hidden_0 = prev_state
        hidden_1 = (torch.zeros_like(hidden_0[0]), torch.zeros_like(hidden_0[1]))

        actions = []
        action_probs = []
        raw_probs = []
        layer_loss = 0
        last_probs = None
        for i in range(self.num_cell):  # decoding all the actions
            inputs = self.embedding(sup_seq[i])
            hidden_0 = self.layer_cell(inputs, hidden_0)
            inputs_1 = hidden_0[0]  # inputs should be the hidden state
            hidden_1 = self.cell(inputs_1, hidden_1)

            output = hidden_1[0]  # output should be the hidden state of the second layer
            raw_prob = self.decoder(output)
        
            raw_prob = torch.tanh(raw_prob)

            raw_probs.append(raw_prob)  # 1 * K
            # last_token = pred
        raw_probs = torch.cat(raw_probs, dim=0)
        sup = torch.cat(sup_seq[1:], dim=0)
        layer_loss = F.cross_entropy(raw_probs, sup)
        return layer_loss, hidden_0


class LogicLayer(nn.Module):
    r"""
    Logic layers are used to maintain the rigger sampler and a set of logic nodes.
    """
    def __init__(
        self, node_num, is_leaf,
        last_layer=None, op_pool=None, type=None, embedding=None, layer_cell=None
    ):
        super().__init__()
        self.is_leaf = is_leaf
        
        if is_leaf:  # Leaf node does not have last layer
            self.nodes = nn.ModuleList(
                [LeafNode(node_idx) for node_idx in range(node_num)]
            )
        else:  # normal logic layer
            self.rigger = RiggerLayer(
                len(last_layer), node_num, len(op_pool), op_pool, embedding, layer_cell
            )
            self.nodes = nn.ModuleList(
                [LogicNode(last_layer, op_pool, type,) for _ in range(node_num)]
            )

    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, idx):
        return self.nodes[idx]

    def sample_action(self, prev_hidden):
        # import pdb; pdb.set_trace()
        actions, action_probs, raw_probs, hidden, layer_loss = self.rigger.sample(prev_hidden)
        for action, prob, raw_prob, node in zip(actions, action_probs, raw_probs, self.nodes):
            size = len(node.last_layer) * len(node.last_layer)
            node.raw_action = action
            node.op_idx = action // size
            action = action % size
            node.node_idx_r = action % len(node.last_layer)
            node.node_idx_l = action // len(node.last_layer)
            node.update_log_probs('prob', prob)
            node.update_log_probs('reward', 0)
            node.update_log_probs('raw_prob', raw_prob)
        return hidden, layer_loss

    def try_to_find_pseudo_label(self):
        assert self.is_leaf
        for key in self.nodes[0].node.keys():
            if self.nodes[0].label != {}:
                for batch_idx in self.nodes[0].node[key].shape[0]:
                    ans = [x.label[key] for x in self.nodes]  # [[], [], []]
                    # prob, ans = zip(ans)
                    # for node_idx in len(self.nodes):
        return None, torch.zeros((10), dtype=torch.bool).cuda()

    def forward(self, last_layer):
        return [node(last_layer) for node in self.nodes]


class DeepLogic(nn.Module):
    r"""
    DeepLogic maintains multiple layers of logic operations.
    A deeplogic model also do two things:
        1) correction the labels from raw inputs (leaf pruning).
        2) sample and update the logic probs of each logic nodes ()
    """
    LOGIC_SYMBOL = {
        'term': TERM_OP,
        'relation': RELATION_OP,
        'relationinit': RELATION_INIT_OP,
        'logic': LOGIC_OP,
    }

    def __init__(self, args, name, inputlayer):
        super().__init__()
        self.name = name
        self.inputlayer = inputlayer
        self.logiclayers = [self.inputlayer]  # inputslayer placeholder
        # self.embedding = nn.Embedding(len(embedding_dict) + 10, embedding_dim=HIDDEN_DIM)
        self.args = args
        shared_layer_cell = nn.LSTMCell(HIDDEN_DIM, HIDDEN_DIM)
        for op_name, node_num in args[1:]:
            current_layer = LogicLayer(
                node_num, False, self.logiclayers[-1], self.LOGIC_SYMBOL[op_name], op_name,
                layer_cell=shared_layer_cell
            )
            self.logiclayers.append(current_layer)
        self.root_node = self.logiclayers[-1][0]
        self.logiclayers = nn.Sequential(*self.logiclayers)

        # Add a container to maintain the best K structure ever found to compute another cul loss
        self.best_models = []
        self.well_trained = False

    def self_evaluate(self):
        if len(self.best_models) > 0:
            visit = [x[2] for x in self.best_models]
            max_id = visit.index(max(visit))
            if self.best_models[max_id][0] > 0.6 and self.best_models[max_id][2] > 10:
                logger.infox("I am well trained!!")
                self.well_trained = True

    def leaf_pruning(self, target, mask, key):
        r"""
        find a valid inputs for the target, max_step is the max number of changed symbol from inputs

        inputs:
            target: the expected results from the logic model.
            max_step: the maximum changes allowed in the input node layer.
            max_width: max_width of each layer used for the beam search algorithm.

        algorithm details:
            label back propagation from root node to leaf node layer by layer.
            In each layer,
                1. check confliction;
                2. gather possibility of the whole layer;
        """
        # async
        def single_instance_single_step_pruning(index, target, key):
            r"""
            index: batch_idx
            target: single value
            """
            # TODO: Consider the searching pruning
            # import pdb; pdb.set_trace()
            if self.root_node.node[key][index] == target:  # does not need correction
                for item in self.inputlayer:
                    item.label[key][index] = (1, item.node[key][index])
                return
            # elif self.root_node.node[key][index] and not target:
            #     for item in self.inputlayer:
            #         d = item.node[key].device
            #         item.label[key][index] = torch.zeros(1, dtype=item.node[key].dtype, device=d)[0]
            #     return False
            queue = Queue()  # A simple FIFO Queue
            queue.put((self.root_node, True, target, "O"))  # node, semaphore, target, path
            # raw_target = target
            while not queue.empty():
                item, semaphore, target, path = queue.get()
                # check leaf node
                if isinstance(item, LeafNode):
                    if semaphore:
                        item.pseudo_label[key][index][path] = target
                    else:
                        item.fixed_value[key][index].add(path)  # fixed by 'path'
                
                    continue  # operation done
                child_l = item.last_layer[item.node_idx_l]
                child_r = item.last_layer[item.node_idx_r]
                if semaphore:
                    p_target_l = item.op_pool[item.op_idx].inv_op(child_r.node[key][index], target)
                    p_target_r = item.op_pool[item.op_idx].inv_op(child_l.node[key][index], target)
                    if p_target_l is not None:
                        for idx, target_l in enumerate(p_target_l):
                            new_path = path + "L%d" % idx
                            queue.put((child_l, True, target_l, new_path))
                            if item.op_pool[item.op_idx] not in UNITARY_OP:
                                queue.put((child_r, False, None, new_path))  # path keeped
                    if item.op_pool[item.op_idx] not in UNITARY_OP and p_target_r is not None:
                    
                        for idx, target_r in enumerate(p_target_r):
                            new_path = path + "R%d" % idx
                            queue.put((child_r, True, target_r, new_path))
                            queue.put((child_l, False, None, new_path))
                else:  # back propagate fixed_value
                    queue.put((child_l, False, None, path))
                    queue.put((child_r, False, None, path))

            # import pdb; pdb.set_trace()
            # find conflict by prefix
            for item in self.inputlayer:
                for forbidden_path in item.fixed_value[key][index]:
                    keys = copy.deepcopy(list(item.pseudo_label[key][index].keys()))
                    for key_ in keys:
                        if key_.startswith(forbidden_path):
                            item.pseudo_label[key][index].pop(key_)  # remove conflict
            # import pdb; pdb.set_trace()
            # find pseudo_label and calculate their prob
            for item in self.inputlayer:
                if item.pseudo_label[key][index]:  # not empty
                    # import pdb; pdb.set_trace()
                    for label_val in item.pseudo_label[key][index].values():
                        item.label[key][index].append(
                            (item.inputs[key][index, label_val] - item.log_prob()[key][index], label_val)
                        )
                        for other in self.inputlayer:
                            if other != item:
                                other.label[key][index].append(
                                    (0, other.node[key][index])
                                )
            # import pdb; pdb.set_trace()
            # maybe empty
            # import pdb; pdb.set_trace()
            # pseudo_labels = zip(*[item.label[key][index] for item in self.inputlayer])
            # # print(list(pseudo_labels))
            # max_log_probs = None
            # best_label = None
            # for item in pseudo_labels:

            #     probs = sum([node[0] for node in item])
            
            #     if max_log_probs is None or probs > max_log_probs:
            #         max_log_probs = probs
            #         best_label = [node[1] for node in item]
            # if best_label:  # find a valid label.
            #     for item, label in zip(self.inputlayer, best_label):
            #         item.label[key][index] = label

            #     return True
            # else:  # empty place holder
            #     for item in self.inputlayer:
            #         item.label[key][index] = torch.zeros(1, dtype=item.node.dtype, device=item.node.device)[0]
            #     return False
        # import pdb; pdb.set_trace()
        # init some empty place holders
        for item in self.inputlayer:
            item.label[key] = [[] for _ in range(len(target))]
            item.pseudo_label[key] = [{} for _ in range(len(target))]
            item.fixed_value[key] = [set() for _ in range(len(target))]

        loop = asyncio.get_event_loop()
        task_list = []
        for batch_index in range(target.shape[0]):
            # if mask[batch_index]:  # mask_dependent or not (From the exp)
            if True:
                single_instance_single_step_pruning(batch_index, target[batch_index], key)
                # task = loop.create_task(
                #     single_instance_single_step_pruning(batch_index, target[batch_index], key)
                # )
                # loop.run_until_complete(task)
                # task_list.append(task)

    def decode_raw_action(self, raw_action, num_inputs):
        size = num_inputs * num_inputs
        op_idx = raw_action // size
        raw_action = raw_action % size
        idx_r = raw_action % num_inputs
        idx_l = raw_action // num_inputs
        return idx_l, idx_r, op_idx

    def vis_sequence(self, best_models=None):
        r"""
            visualize sequences without the help of iteration
        """
        if best_models is None:
            best_models = self.best_models

        for reward, model, visit in best_models:
            layer_strs = ['[%d]' % d for d in range(self.args[0][1])]  # input_layers
            for model_layer, (op_name, _) in zip(model, self.args[1:]):
                new_layer = []
                for model_node in model_layer:
                    idx_l, idx_r, idx_op = self.decode_raw_action(model_node, len(layer_strs))
                    if self.LOGIC_SYMBOL[op_name][idx_op] in UNITARY_OP:
                        new_layer.append(
                            "{}({})".format(
                                self.LOGIC_SYMBOL[op_name][idx_op].__class__.__name__,
                                layer_strs[idx_l]
                            )
                        )
                    else:
                        new_layer.append(
                            '{}({}, {})'.format(
                                self.LOGIC_SYMBOL[op_name][idx_op].__class__.__name__,
                                layer_strs[idx_l],
                                layer_strs[idx_r]
                            )
                        )
                layer_strs = new_layer
            logger.info("REWARD: {}; Visit: {}, model:".format(reward, visit))
            for layer_str in layer_strs:
                logger.info(" " * 5 + layer_str + '\n')

    def equiv_check(self, model_a, model_b):
        def str2tree(model):
            layer_strs = [d for d in range(self.args[0][1])]  # input_layers
            for model_layer, (op_name, _) in zip(model, self.args[1:]):
                new_layer = []
                for model_node in model_layer:
                    idx_l, idx_r, idx_op = self.decode_raw_action(model_node, len(layer_strs))
                    if self.LOGIC_SYMBOL[op_name][idx_op] in UNITARY_OP:
                        new_layer.append(
                            (self.LOGIC_SYMBOL[op_name][idx_op].__class__.__name__,
                             layer_strs[idx_l])
                        )
                    else:
                        new_layer.append(
                            (self.LOGIC_SYMBOL[op_name][idx_op].__class__.__name__,
                             layer_strs[idx_l],
                             layer_strs[idx_r])
                        )
                layer_strs = new_layer

            return layer_strs

        tree_a = str2tree(model_a)
        tree_b = str2tree(model_b)

        def check_eq(sub_tree_a, sub_tree_b):

            if not isinstance(sub_tree_a, tuple) or not isinstance(sub_tree_b, tuple):
                return sub_tree_a == sub_tree_b
            if len(sub_tree_a) != len(sub_tree_b) or sub_tree_a[0] != sub_tree_b[0]:
                return False  # dif len or dif op
            if len(sub_tree_a) == 2 and check_eq(sub_tree_a[1], sub_tree_b[1]):
                return True
            elif len(sub_tree_a) == 3:
                return ((check_eq(sub_tree_a[1], sub_tree_b[1])
                         and check_eq(sub_tree_a[2], sub_tree_b[2]))
                        or (check_eq(sub_tree_a[2], sub_tree_b[1])
                            and check_eq(sub_tree_a[1], sub_tree_b[2])
                            and True))  # Left placeholder for asymmetric op
        # import pdb; pdb.set_trace()
        return reduce(
            lambda x, y: x and y,
            [check_eq(a, b) for a, b in zip(tree_a, tree_b)]
        )
        # eq = check_eq(tree_a, tree_b)
        # if eq:
        #     logger.info("Find Eq: {}, {}".format(tree_a, tree_b))
        # return eq

    def sample_actions(self):
        layer_loss = 0
        hidden = (
            torch.zeros(size=(1, HIDDEN_DIM)).cuda(),  # batch size is always 1 for logic sampling
            torch.zeros(size=(1, HIDDEN_DIM)).cuda()
        )  # H_0, C_0
        for layer in self.logiclayers[1:]:
            hidden, layer_loss_term = layer.sample_action(hidden)
            layer_loss += layer_loss_term

        # logger.info(self.root_node.vis_connection(0))
        return layer_loss

    def apply_best_model(self):
        visit = [x[2] for x in self.best_models]
        max_id = visit.index(max(visit))
        best_model = self.best_models[max_id][1]
        logger.info("Apply current best model and fix the structure:")
        self.vis_sequence([self.best_models[max_id]])
        for layer_action, layer in zip(best_model, self.logiclayers[1:]):
            for action, node in zip(layer_action, layer): 
                size = len(node.last_layer) * len(node.last_layer)
                node.raw_action = action
                node.op_idx = action // size
                action = action % size
                node.node_idx_r = action % len(node.last_layer)
                node.node_idx_l = action // len(node.last_layer)

    def best_model_tracing(self, reward):
        r"""
        reward: a number that evaluates current structure
        """
        current_model = [
            [node.raw_action for node in layer] for layer in self.logiclayers[1:]
        ]
        for idx, (r, m, c) in enumerate(self.best_models):

            if self.equiv_check(m, current_model):  # consider the equivalence
                # import pdb; pdb.set_trace()
                new_reward = (r * c + reward) / (c + 1)  # mean
                # import pdb; pdb.set_trace()
                self.best_models[idx] = (
                    new_reward, current_model, c + 1
                )
                heapq.heapify(self.best_models)
                break
        else:  # if not break
            if reward > 0.6:
                heapq.heappush(self.best_models, (reward, current_model, 1))
        if len(self.best_models) > 0 and self.best_models[0][0] < 0.50:
            logger.info("Pop an instance due to low reward:{}".format(self.best_models[0][0]))
            self.vis_sequence([self.best_models[0]])
            heapq.heappop(self.best_models)
        if len(self.best_models) > 10:
            heapq.heappop(self.best_models)  # remove the one with smallest reward

        if len(self.best_models) > 2:
            # self.vis_sequence(self.best_models)
            probs = [x[0] for x in self.best_models]  # according to the minimal prob
            # action = random.randint(range(len(self.best_models)), probs=probs)
            action = Categorical(logits=torch.Tensor(probs).cuda() * 10).sample()
            # logger.info(action)
            best_model = self.best_models[action][1]
            pseudo_label_loss = 0
            hidden = (
                torch.zeros(size=(1, HIDDEN_DIM)).cuda(),  # batch size is always 1 for logic sampling
                torch.zeros(size=(1, HIDDEN_DIM)).cuda()
            )  # H_0, C_0
            for layer_action, layer in zip(best_model, self.logiclayers[1:]):
                loss_term, hidden = layer.rigger.supervised_loss(hidden, layer_action)
                pseudo_label_loss += loss_term
                # for node_action, node in zip(layer_action, layer):
                #     pseudo_label_loss += F.cross_entropy(node._log_probs['raw_prob'], node_action)
            return pseudo_label_loss
        else:
            return 0

    def reward_backpropagate(self, reward):
        
        queue = Queue()
        queue.put((self.root_node, reward))
        # for node in self.root_nodes:
        #     queue.put((node, reward))

        # reward backpropagate
        while not queue.empty():
            node, reward = queue.get()
            if isinstance(node, LeafNode): continue  # omit leaf node

            node._log_probs['reward'] += reward  # reward accumulate
            queue.put((node.last_layer[node.node_idx_l], reward))
            if node.op_pool[node.op_idx] not in UNITARY_OP:
                queue.put((node.last_layer[node.node_idx_r], reward))

    def rl_loss_gathering(self):
        # reward accumulate
        loss = 0
        for layer in self.logiclayers[1:]:
            for node in layer:
                loss -= node._log_probs['prob'] * node._log_probs['reward']
                node._log_probs['prob'] = 0
                node._log_probs['reward'] = 0
        return loss

    def vis_connection(self):
        print(self.root_node.vis_connection(0))

    def forward(self, symbols, sym_id=''):
        r"""
        symbols: B, L, N
        """
        res = self.logiclayers([symbols, sym_id])  # forward pass
        return res[0][0]
