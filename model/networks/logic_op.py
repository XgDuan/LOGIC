r"""
Logic Definition. In this module, we define basic logic operations.

Basically, we define the following logics:
    1) And
    2) Or
    3) Ge
    4) Le
    5) Not
    6) KeepFirst

Ref to this paper: CLN2INV: LEARNING LOOP INVARIANTS WITH CONTINUOUS LOGIC NETWORKS
Firstly, we consider some logical operations that convert logical values into new logical values.
    And, Or, Not

Also, we should consider operations that convert real values into bool value (Ge, Le).

Besides, the above operations convert real values into a bool value (Ge, Le). We should also have
    another type of operations that convert real values into another real values (Add, Sub).

But, the above setting would make the whole system quite complex.

Good idea is: The above setting fits the definition in FOL as predictive, relation, and special symb

Moreover,

"""
from itertools import chain

from numpy.core.fromnumeric import prod
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.activation import LogSigmoid


# class Not:
#     def __call__(self, first, second):
#         return 1 - first

####################################################################################################
# Lukasiewicz
####################################################################################################


class Land:
    def __call__(self, first, second):
        return torch.max(0, first + second - 1)


class Lor:
    def __call__(self, first, second):
        return torch.min(1, first + second)


####################################################################################################
# Godel
####################################################################################################


class Gand:
    def __call__(self, first, second):
        return torch.min(first, second)


class Gor:
    def __call__(self, first, second):
        return torch.max(first, second)


####################################################################################################
# Product
####################################################################################################


class Pand:
    def __call__(self, first, second):
        return first * second


class Por:
    def __call__(self, first, second):
        return first + second - first * second


####################################################################################################
# Other Ops that are differentable
####################################################################################################
class Add:
    def __call__(self, t1, t2):
        r"""
        Here t1 and t2 are two addable terms, and this function returns the added results of them
        """
        sum_ = t1 + t2
        return sum_ + (sum_ % 10 - sum_).detach()

    def inv_op(self, one_op, target):
        res = target - one_op
        return [res] if res > -1 else None


# Relation Symbols: R(t_1, t_2, ...t_n) \to True / False
class Eq:

    B, epsilon = 1, 1e-3

    def __call__(self, t1, t2):
        return t1 == t2

    def inv_op(self, one_op, target):
        if target:
            return [one_op]
        else:
            return [(one_op - i) % 10 for i in range(1, 10)]


class Keep:
    def __call__(self, t1, t2):
        r"""
        """
        return t1

    def inv_op(self, op, target):
        return [target]


# Definition of operations:
# TERM_OP = [
#     Add(), Sub(), Prod(), Keep(),
# ]
# RELATION_INIT_OP = [
#     Eq(), Neq()
# ]

class And:
    def __call__(self, l1, l2):
        return torch.logical_and(l1, l2)

    def inv_op(self, one_op, target):
        if one_op:  # one_op is true, the other shoule be the same with target
            return [target]
        elif target:  # one_op false, target_true, return None
            return None
        else:
            return [one_op, target]


class Or:
    def __call__(self, l1, l2):
        return torch.logical_or(l1, l2)

    def inv_op(self, one_op, target):
        if not one_op:
            return [target]
        elif target:
            return [target, not target]
        else:
            return None


class Not:
    def __call__(self, l1, l2):
        return torch.logical_not(l1)

    def inv_op(self, op, target):
        return [not target]


TERM_OP = [
    Add(), Keep()
]
RELATION_INIT_OP = [
    Eq()
]
RELATION_OP = [
    RELATION_INIT_OP[0], Keep()
]
LOGIC_OP = [
    And(), Or(), Keep(), Not()
]

UNITARY_OP = [
    TERM_OP[1], RELATION_OP[1], LOGIC_OP[2], LOGIC_OP[3]
]

embedding_dict = {
    x: idx for idx, x in enumerate(chain(TERM_OP, RELATION_OP, LOGIC_OP))
}
embedding_dict['BOS'] = len(embedding_dict)