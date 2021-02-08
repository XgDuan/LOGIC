r"""
The definition of the basic logic layer.
A logic layer is a binary layer that the output is the logical combination of the input layer
"""

# Function Symbols: f(t_1, t_2, t_3,...t_n) \to t_{new}
# In this paper we focus on those function symbols with 2-arity, i.e. n=2

import torch
import torch.nn as nn


class Add:
    def __call__(self, t1, t2):
        r"""
        Here t1 and t2 are two addable terms, and this function returns the added results of them
        """
        return t1 + t2


class Sub:
    def __call__(self, t1, t2):
        r"""
        Here t1 and t2 are two addable terms, and this function returns their difference
        """
        # TODO: Maybe we should filter the results here (filter out those negative one)
        return t1 - t2


class Prod:
    def __call__(self, t1, t2):
        r"""
        return t1 * t2
        """
        return t1 * t2


# Relation Symbols: R(t_1, t_2, ...t_n) \to True / False

class Eq:
    def __call__(self, t1, t2):
        return (t1 == t2).float()
        return t1 == t2


class Neq:
    def __call__(self, t1, t2):
        return (t1 == t2).float()
        return t1 != t2


class And:
    def __call__(self, f1, f2):
        r"""
        f1, f2 are two formulars
        """
        return torch.bitwise_and(f1, f2)


class Or:
    def __call__(self, f1, f2):
        return torch.bitwise_or(f1, f2)


class Not:
    def __call__(self, f1, f2):
        r"""
        Here we treat f2 as a dummy input
        """
        return torch.bitwise_not(f1)


class Keep:
    def __call__(self, f1, f2):
        r"""
        """
        return f1


# Definition of operations:
TERM_OP = [
    Add(), Sub(), Prod(), Keep(),
]
RELATION_INIT_OP = [
    Eq(), Neq()
]
RELATION_OP = [
    Eq(), Neq(), Keep()
]
LOGIC_OP = [
    And(), Or(), Not(), Keep()
]


# Selectors
class GumbelSoftmaxSelector(nn.Module):
    def __init__(self, dim=0, interval=100, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.dim = dim
        self.softmax = nn.Softmax(dim=self.dim)
        self.anneal_rate = 0.00003
        self.interval = 100
        self.counter = 0
        self.temperature_min = 0.5

    # def anneal(self):
    #     self.temperature = max(
    #         self.temperature * math.exp(-self.anneal_rate * self.counter), self.temperature_min
    #     )

    def sample_gumbel(self, logits, eps=1e-20):
        U = torch.rand_like(logits)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits):
        y = logits + self.sample_gumbel(logits)
        return self.softmax(y / self.temperature)

    def forward(self, logits):
        self.counter += 1
        if self.counter % self.interval == self.interval - 1:
            self.anneal()
        y = self.gumbel_softmax_sample(logits)
        _, ind = y.max(dim=self.dim)
        y_hard = torch.zeros_like(y)
        # import pdb; pdb.set_trace()
        y_hard.scatter_(self.dim, ind.unsqueeze(self.dim), 1)
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def __repr__(self):
        return "{}(dim={})".format(
            self.__class__.__name__, self.dim)


class SoftmaxSelector(nn.Module):
    def __init__(self, num_input, num_output, temperature_min=0.2, temperature=1):
        super().__init__()
        self.parameter = nn.Parameter(
            torch.randn(num_output, num_input),
            requires_grad=True,
        )
        self.temperature_min = temperature_min
        self.temperature = temperature
        self.dim = 1
        self.softmax = nn.Softmax(self.dim)

    # def anneal(self):
    #     self.temperature = max(
    #         self.temperature * math.exp(-self.anneal_rate * self.counter), self.temperature_min
    #     )

    def forward(self, inputs):
        r"""
        inputs: B * N
        """
        y = self.softmax(self.parameter)  # maybe we do not need this normalization
        y_max, ind = y.max(dim=self.dim)  # M
        y_hard = (y_max - y_max.detach()) + 1  # HARD MAX LAZY IMPLEMENTATION
        outputs = inputs.index_select(dim=self.dim, index=ind)  # B * M
        return outputs * y_hard

    def __repr__(self):
        return "{}(dim={})".format(
            self.__class__.__name__, self.dim)


# More details of logics. We will define basic layers, and the selectors.

class BasicLogicLayer(nn.Module):
    def __init__(self, selector, op_list):
        super().__init__()
        self.selector = selector
        self.op_list = op_list
    
    def forward(self, inputs):
        r"""
        Enumerate all the possible logic operations and return a small subset of logics that are
            more possible to be useful.

        inputs: B * K * N; B: batch size; K: num of groups; N: number of objects.
            terms in the same group share the same selector.
        outputs: B * K * M; B: batch size; K: num of groups; M: number of outputs.
        """
        
        # TODO: test the speed of uptriangle matrix and full matrix
        B, K, N = inputs.shape
        tui_indices = torch.triu_indices(N, N, 1).to(inputs.device)
        tui_indices = tui_indices[None, None, :, :].expand(B, K, -1, -1)
        inputs_first = inputs.gather(dim=2, index=tui_indices[:, :, 0, :])
        inputs_second = inputs.gather(dim=2, index=tui_indices[:, :, 1, :])

        res = [op(inputs_first, inputs_second) for op in self.op_list]
        res = torch.cat(res, dim=-1)  # cat these results in the last dimension

        # TODO: tracing the selected operations
        selected_res = self.selector(res)

        return selected_res


class TermLayer(BasicLogicLayer):
    def __init__(self, selector, func_list):
        super().__init__(selector, func_list)


class RelationLayer(BasicLogicLayer):
    def __init__(self, selector, relation_list):
        super().__init__(selector, relation_list)


class LogicLayer(BasicLogicLayer):
    def __init__(self, selector, logic_list):
        super().__init__(selector, logic_list)


class DeepLogic(nn.Module):
    def __init__(
        self, cfg, name='logits',
        func_list=(Add, Sub, Prod),
        relation_list=(Eq, Neq),
        logic_list=(And, Or, Not)
    ):
        super().__init__()

        self.terms = nn.Sequential(
            *[TermLayer(SoftmaxSelector(i, o), func_list)
              for i, o in zip(cfg.term_inputs, cfg.term_outputs)]
        )
        self.relations = nn.Sequential(
            *[RelationLayer(SoftmaxSelector(i, o), relation_list)
              for i, o in zip(cfg.rel_inputs, cfg.rel_outputs)]
        )
        self.logics = nn.Sequential(
            *[LogicLayer(SoftmaxSelector(i, o), logic_list)
              for i, o in zip(cfg.logic_inputs, cfg.logic_outputs)]
        )

        self.repr_str = "{}(name={})".format(self.__class__.__name__, name)
    
    def __repr__(self):
        return self.repr_str

    def forward(self, inputs):
        r"""
        inputs: B *  N
        """
        return self.logics(self.relations(self.terms(inputs)))
