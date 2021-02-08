r"""
Generating the attributes and rules pairs
"""
import argparse
import random
import itertools
from functools import reduce

import numpy as np

from .const import RULES, ATTRIBUTES


def enumerate_attr_rule_combinations(noisy_rules=False):
    r"""
    Find all the possible attribute-rule combinations.
    """
    possible_attr_rule_combinations = [
        [(attr_name, rule_name, rule_val)
            for rule_name in RULES.keys()
            for rule_val in RULES[rule_name]
         ] for attr_name in ATTRIBUTES.keys()
    ]

    # Random placeholder here
    random_attr_rule_placeholder = [
        [(attr_name, "RANDOM", 0)] for attr_name in ATTRIBUTES.keys()
    ]

    # #{All combinations} < 480 * 1000
    if noisy_rules:
        return itertools.chain(
            itertools.product(
                possible_attr_rule_combinations[0],
                possible_attr_rule_combinations[1],
                random_attr_rule_placeholder[2]
            ),
            itertools.product(
                possible_attr_rule_combinations[0],
                random_attr_rule_placeholder[1],
                possible_attr_rule_combinations[2]
            ),
            itertools.product(
                random_attr_rule_placeholder[0],
                possible_attr_rule_combinations[1],
                possible_attr_rule_combinations[2]
            ),
        )
    return itertools.product(*possible_attr_rule_combinations)


def enumerate_attr_rule_instances(seq_len=3, ring=True):
    r"""
    Generate all possible pairs of attribute-rule instances.
    Return:
        {
            "ATTR_NAME": {
                "RULE_NAME": {
                    "RULE_VALUE": [
                        [attr_values] * seq_len,
                        [attr_values]
                    ]
                }
            }
        }
    """
    # res = h5py.File("Test.hdf5", "w")
    # for attr_k, attrs in ATTRIBUTES.items():
    #     res.create_group(attr_k)
    #     for rule_k, rules in RULES.items():
    #         res[attr_k].create_group(rule_k)
    RULES["RANDOM"] = [0]
    res = {}
    for attr_k, attrs in ATTRIBUTES.items():
        res[attr_k] = {}
        for rule_k, rules in RULES.items():
            res[attr_k][rule_k] = {}

    def prod(ll):
        return reduce(lambda x, y: x * y, ll)

    def handle_arithmetric(attr_cand, step):
        first_n = itertools.product(attr_cand, repeat=seq_len - 1)
        res = []
        for possible in first_n:
            if step == "ADD" and (ring or sum(possible) <= attr_cand[-1]):
                res.append([*possible, sum(possible) % (attr_cand[-1] + 1)])
            elif step == "PROD" and (ring or prod(possible) <= attr_cand[-1]):
                res.append([*possible, prod(possible) % (attr_cand[-1] + 1)])
        return res

    def handle_progression(attr_cand, step):
        res = []
        for first in attr_cand:
            current = first
            instance = [current]
            for possible in range(seq_len - 1):
                current += step
                if len(attr_cand) == 10 and \
                   (step > 0 and current > attr_cand[-1] or step < 0 and current < attr_cand[0]):
                    break
                instance.append(current % (attr_cand[-1] + 1))
            if len(instance) < seq_len:
                continue
            res.append(instance)
        return res
    
    def handle_const(attr_cand, step):
        res = []
        for attr in attr_cand:
            res.append([attr] * seq_len)
        return res
    
    def handle_dist_three(attr_cand, step):
        return list(itertools.permutations(attr_cand, seq_len))

    def handle_random(attr_cand, step):
        res = []
        for _ in range(1000):
            res.append(random.sample(attr_cand, seq_len))
        return res

    handles = {
        "ARITHMETRIC": handle_arithmetric,
        "PROGRESSION": handle_progression,
        "CONST": handle_const,
        "DIST_THREE": handle_dist_three,
        "RANDOM": handle_random
    }
    for attr_k, attrs in ATTRIBUTES.items():
        for rule_k, rules in itertools.chain(RULES.items()):   # Append random rule
            for rule_v in rules:
                results = handles[rule_k](attrs, rule_v)
                res[attr_k][rule_k][rule_v] = results
                # res[attr_k][rule_k].create_dataset(str(rule_v), data=results)
    return res


def dataset_generation(dataset_size=10000, seq_len=3):
    r"""
    Each instance in the dataset includes three lines:
        the first two lines are ground truth sequence strictly following the hidden rules
        while the third line missed the last entity.
    """
    attr_rule_comb = list(enumerate_attr_rule_combinations())
    attr_rule_inst = enumerate_attr_rule_instances(seq_len=seq_len)
    # import pdb; pdb.set_trace()
    res = []

    def generate_wrong_ans(attr_rule, attr_matrix, wa_num=7):
        r"""
        Here we  generate wrong  answers for the task.
        TODO:  considering  better  wrong  answer  generator
        """
        third_row = attr_matrix[:, 6:]  # K, 3
        answer = attr_matrix[:, -1]  # K
        wa = np.repeat(answer[:, np.newaxis], wa_num, axis=1)  # K * wa_num
        for idx, attr_rule_val in enumerate(attr_rule):
            if attr_rule_val[1] == "DIST_THREE":  # Make the third answer the same with other
                wa[idx, :wa_num // 2] = third_row[idx, 0]  # Half as the first
                wa[idx, wa_num // 2:] = third_row[idx, 1]  # Half as the second
            else:  # Else random
                wa[idx] = np.arange(wa_num) - wa_num // 2
                wa[idx, : wa_num // 2 + 1] = wa[idx, : wa_num // 2 + 1] - 1
                wa[idx] = (wa[idx] + answer[idx]) % (len(ATTRIBUTES[attr_rule_val[0]]) + 1)
                wa = wa % (ATTRIBUTES[attr_rule_val[0]][-1] + 1)
        return wa

    for _ in range(dataset_size):
        ar = random.choice(attr_rule_comb)  # sample the attr_rule
        # import pdb; pdb.set_trace()
        attr_list = [
            random.sample(attr_rule_inst[ar[i][0]][ar[i][1]][ar[i][2]], 3)
            for i in range(len(ar))
        ]  # [[[], [], []], [[], [], []], [[], [], []]]
        attr_matrix = np.array(attr_list).reshape(len(attr_list), -1)
        wrong_ans = generate_wrong_ans(ar, attr_matrix, 7)
        res.append((attr_matrix, wrong_ans))
    return res


def argument_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--")

    return parser


def main():
    res = dataset_generation(100)
    for pairs in res:
        print(pairs)
    # print(res)


if __name__ == "__main__":
    main()
