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


from numpy.core.fromnumeric import prod
import torch
import torch.nn as nn
from torch.autograd import Function


class Not:
    def __call__(self, first, second):
        return 1 - first

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
        return sum_
        # return sum_ + (sum_ % 1 - sum_).detach()


class Sub:
    def __call__(self, t1, t2):
        r"""
        Here t1 and t2 are two addable terms, and this function returns their difference
        """
        # TODO: Maybe we should filter the results here (filter out those negative one)
        dif_ = t1 - t2
        return dif_
        return dif_ + (dif_ % 1 - dif_).detach()


class Prod:
    def __call__(self, t1, t2):
        r"""
        return t1 * t2
        """
        prod_ = t1 * t2
        return prod_  # prod won't exceed the limits


# Relation Symbols: R(t_1, t_2, ...t_n) \to True / False

class Eq:
    B, epsilon = 1, 1e-3

    def __call__(self, t1, t2):
        
        dist = torch.sum((t1 - t2) ** 2, dim=1, keepdim=True)
        # print(dist.mean())
        return torch.exp(- 0.1 * dist)
        # inner_prod = torch.sum(t1 * t2, dim=1, keepdim=True)  # Batch, 1
        # abs_prod = torch.sum(torch.abs(t1) * torch.abs(t2), dim=1, keepdim=True)
        # return (inner_prod / (abs_prod + 1e-5) + 1) / 2
        # return 1 / ((1 + torch.exp(self.B * (t1 - t2 - self.epsilon))) * (1 + torch.exp(- self.B * (t1 - t2 + self.epsilon))))
        # # return 1 - (t1 - t2).abs()   # remember the t-norm!!!


class Neq:
    def __call__(self, t1, t2):
        return (t1 - t2).abs()


class Keep:
    def __call__(self, t1, t2):
        r"""
        """
        return t1


# Definition of operations:
# TERM_OP = [
#     Add(), Sub(), Prod(), Keep(),
# ]
# RELATION_INIT_OP = [
#     Eq(), Neq()
# ]
TERM_OP = [
    Add(), Keep()
]
RELATION_INIT_OP = [
    Eq()
]
RELATION_OP = [
    Eq(), Neq(), Keep()
]
LOGIC_OP = [
    Pand(), Por(), Not(), Keep()
]