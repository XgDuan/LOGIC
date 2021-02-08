import torch
import torch.nn as nn
import torch.distributions as distributions
from torch.autograd import Function
from .networks import CNN, MLP, Symbolizer
from .networks.logic_op import TERM_OP, RELATION_OP, RELATION_INIT_OP, LOGIC_OP

from .basic_model import BasicModel

r"""
In this function, we define the basic

"""


class LogicNodeFunction(Function):
    @staticmethod
    def forward(ctx, input_1, input_2, operation_logits, operation_pool, sample_path=True):
        # save the log prob of operation and input_1, input_2?
        # TODO: using the default distributions.Categorical (considering the gumbel-max trick later)
        sampler = distributions.Categorical(logits=operation_logits)
        # whether to sample the symbhols or not
        if sample_path:
            operation_idx = sampler.sample()
        else:
            _, operation_idx = operation_logits.max(dim=-1)
        log_probs = sampler.log_prob(operation_idx)
        operation = operation_pool[operation_idx]
        # import pdb; pdb.set_trace()
        ouput = operation(input_1, input_2)
        ctx.save_for_backward(operation_logits, operation_idx, log_probs)
        return ouput

    @staticmethod
    def backward(ctx, grad_output):
        operation_logits, operation_idx, log_probs = ctx.saved_tensors
        da_dl = torch.softmax(operation_logits, dim=-1)
        # import pdb; pdb.set_trace()
        operation_idx = operation_idx.unsqueeze(dim=-1)
        do_da = grad_output.mean(dim=0).unsqueeze(dim=-1)
        da_dl.scatter_add_(dim=-1, index=operation_idx, src=-torch.ones_like(operation_idx).float())
        do_dl = do_da * da_dl
        return grad_output, grad_output, do_dl, None, None


class LogicNode(nn.Module):
    def __init__(self, num_input, operation_pool, type, sample_path=True):
        super().__init__()

        # Each node has its own states to recode:
        #   two input index, and its operation, and current reward state
        # Each node is also able to has access to its children's states
        # TODO: Considering the matrix form of the whole layer of logic nodes
        self.logits = torch.nn.parameter.Parameter(torch.zeros(len(operation_pool)))
        self.operation_pool = operation_pool
        self.num_input = num_input
        self.logic_type = type
        self.index_1 = torch.randint(num_input, (1, ))  # RANDOM INIT
        self.index_2 = torch.randint(num_input, (1, ))  # RANDOM INIT
        self.sample_path = sample_path

        # Input index selection
    def __repr__(self):
        return "{}(num_input={}, logic_type={}, logic_op=[{}])".format(
            self.__class__.__name__, self.num_input, self.logic_type,
            ", ".join([x.__class__.__name__ for x in self.operation_pool]))

    def forward(self, input_1, input_2):
        return LogicNodeFunction.apply(
            input_1, input_2, self.logits, self.operation_pool, self.sample_path
        )


class LogicLayer(nn.Module):
    def __init__(self, node_num_i, node_num, operation_pool, type):
        super().__init__()
        self.nodes = nn.ModuleList(
            [LogicNode(node_num_i, operation_pool, type) for _ in range(node_num)]
        )
        self.type = type
        self.operation_pool = operation_pool

    def forward(self, symbol_pool):
        results_symbol = []
        for node in self.nodes:
            input_first = symbol_pool[:, node.index_1]
            input_second = symbol_pool[:, node.index_2]
            results_symbol.append(node.forward(input_first, input_second))
        return torch.cat(results_symbol, dim=1)


class DeepLogic(nn.Module):
    LOGIC_SYMBOL = {
        'term': TERM_OP,
        'relation': RELATION_OP,
        'relationinit': RELATION_INIT_OP,
        'logic': LOGIC_OP,
    }

    def __init__(self, args, name):
        super().__init__()
        self.name = name
        self.logiclayers = nn.ModuleList(
            [LogicLayer(node_num_i, node_num, self.LOGIC_SYMBOL[operation_name], operation_name)
             for operation_name, node_num_i, node_num in args
             ]
        )

    def forward(self, symbols):
        for layer in self.logiclayers:
            symbols = layer(symbols)
        return symbols


class LogicNet(BasicModel):
    def __init__(self, args):
        super().__init__(args)
        self.encoder = CNN(args.encoder)
        # import pdb; pdb.set_trace()
        self.symbolizers = torch.nn.ModuleDict(
            {
                key: torch.nn.Sequential(MLP(args.encoder.output_size, val.num_output, name=key),
                                         Symbolizer(val.sample_path))
                for key, val in args.symbolizer.items()
            }
        )
        self.deeplogic_dict = torch.nn.ModuleDict(
            {key: DeepLogic(val, name=key) for key, val in args.deeplogic.items()}
        )
    
    def compute_loss(self, image, target, meta_data=None):
        pred = self.forward(image)
        loss = torch.sum((pred == target.unsqueeze(1).long()))
        return loss, pred, ''

    def forward(self, inputs, aux=None):
        """
        inputs: B * K * C * H * W (image shape)
        """
        B, K, C, H, W = inputs.shape
        cnn_feature = self.encoder(inputs.view(B * K, C, H, W)).view(B, K, -1)
        symbols = [symbolizer(cnn_feature) for symbolizer in self.symbolizers.values()]
        results = [deeplogic(symbol)
                   for symbol, deeplogic in zip(symbols, self.deeplogic_dict.values())]
        return results[0]
