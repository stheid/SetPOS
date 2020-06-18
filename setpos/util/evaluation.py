from collections import OrderedDict
from operator import itemgetter
from typing import Union

import numpy as np


def create_g_alpha_beta(a: object, b: object, K: object) -> object:
    return lambda s: 1 - a * ((s - 1) / (K - 1)) ** b


def create_g_delta_gamma(delta, gamma):
    return lambda s: delta / s - gamma / s ** 2


def create_g_risk_averse(aversion):
    return lambda s: 1 - np.exp(-aversion / s)


def util(g, target, pred_set):
    return score(target, pred_set) * g(len(pred_set - target.keys()) + 1)


def score(targets: dict, pred: Union[set, str]):
    if isinstance(pred, str):
        pred = {pred}
    score = sum([targets[tag] for tag in pred if tag in targets.keys()])
    total = sum(targets.values())
    score = score / total
    if 1 < total:
        score **= 2
    return score


def create_ubop_predictor(util, is_weigthed=False):
    def ubop_predictor(proba: np.array, labels: tuple, islogproba=False):
        best_set, curr_set, curr_proba, best_util = {}, set(), 0, 0
        sorted_proba = sorted(zip(np.exp(proba) if islogproba else proba, labels))
        while sorted_proba:
            # probabilities sorted ascending, therefore we pop the last element as highest proba
            proba, label = sorted_proba.pop()
            curr_set.add(label)
            curr_proba += proba
            curr_util = curr_proba * util(len(curr_set))
            if best_util < curr_util:
                best_set[label] = proba
                best_util = curr_util
            else:
                break
        if not is_weigthed:
            return set(best_set.keys())
        return OrderedDict(sorted(best_set.items(), key=itemgetter(1), reverse=True))

    return ubop_predictor
