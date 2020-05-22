from itertools import *

import numpy as np
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Hyperparameter, Constant


def powerset(iterable, k=4):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return tuple(chain.from_iterable(combinations(s, r) for r in range(min(k, len(s)) + 1)))


def add_optional_hyperparameter(self: ConfigurationSpace, hyperparam: Hyperparameter, is_enabled_default=False):
    enable_param = CategoricalHyperparameter(hyperparam.name + '__enable', [True, False],
                                             default_value=is_enabled_default)
    self.add_hyperparameters([
        enable_param,
        hyperparam
    ])
    self.add_condition(
        EqualsCondition(hyperparam, enable_param, True)
    )
    return hyperparam


ConfigurationSpace.add_optional_hyperparameter = add_optional_hyperparameter

cs = ConfigurationSpace()

arch_prefix = 'corenlp_train_params__arch__'
cs.add_hyperparameter(CategoricalHyperparameter(arch_prefix + f'wordTag__{1}', [(0, 1)], default_value=(0, 1)))
cs.add_hyperparameter(CategoricalHyperparameter(arch_prefix + 'words', [(-3, 2)], (-3, 2)))

cs.add_hyperparameter(Constant(arch_prefix + 'tag_window_offset', -2))
cs.add_hyperparameter(CategoricalHyperparameter(arch_prefix + 'order', [(0, 2)], (0, 2)))
cs.add_hyperparameter(CategoricalHyperparameter(arch_prefix + f'prefix__multi', [((1, 0), (2, 0), (3, 0))],
                                                default_value=((1, 0), (2, 0), (3, 0))))
cs.add_hyperparameter(CategoricalHyperparameter(arch_prefix + f'suffix__multi', [((2, 0), (3, 0), (4, 0), (5, 0))],
                                                default_value=((2, 0), (3, 0), (4, 0), (5, 0))))

# w_low, w_up = -3, 3
# w_default_range = range(w_low, w_up + 1)
# wchoices = [(i, j) for i in w_default_range for j in range(i + 1, w_up + 1)]
# xfix_range = range(1, 4)
# t_limit = 3
# tchoices = [(i, j) for i in range(t_limit + 1) for j in range(i + 1, t_limit + 1)]
# arch_prefix = 'corenlp_train_params__arch__'

# cs.add_hyperparameter(CategoricalHyperparameter(arch_prefix + 'words', wchoices, (-2, 2)))
# cs.add_hyperparameter(CategoricalHyperparameter(arch_prefix + 'words', [(-2, 2)], (-2, 2)))

# cs.add_optional_hyperparameter(CategoricalHyperparameter(arch_prefix + 'biwords', wchoices, (-1, 1)))
# cs.add_optional_hyperparameter(CategoricalHyperparameter(arch_prefix + 'lowercasewords', wchoices, (-1, 1)))

# cs.add_hyperparameter(UniformIntegerHyperparameter(arch_prefix + 'tag_window_offset', -2, 1, default_value=-2))
# cs.add_optional_hyperparameter(CategoricalHyperparameter(arch_prefix + 'tags', tchoices, (0, 2)))
# cs.add_optional_hyperparameter(CategoricalHyperparameter(arch_prefix + 'twoTags', tchoices, (0, 2)))
# cs.add_optional_hyperparameter(CategoricalHyperparameter(arch_prefix + 'treeTags', tchoices, (0, 2)))
# cs.add_hyperparameter(CategoricalHyperparameter(arch_prefix + 'order', tchoices, (0, 2)))
# cs.add_optional_hyperparameter(CategoricalHyperparameter(arch_prefix + 'order', tchoices, (0, 2)), True)
# cs.add_hyperparameter(Constant(arch_prefix + 'order', 2))
# for i, (word, tag) in enumerate(product(w_default_range, range(t_limit + 1))):
#     cs.add_hyperparameter(
#         CategoricalHyperparameter(arch_prefix + f'wordTag__{i}', [np.NaN, (word, tag)], default_value=np.NaN))
#     for tag2 in range(tag + 1, t_limit + 1):
#         cs.add_hyperparameter(
#             CategoricalHyperparameter(arch_prefix + f'wordTwoTags__{i}{tag2:02d}', [np.NaN, (word, tag, tag2)],
#                                       default_value=np.NaN))

# small_wrange = range(w_low + 1, w_up)
# cs.add_hyperparameter(CategoricalHyperparameter(
#     arch_prefix + f'prefixsuffix__multi', powerset(xfix_range), default_value=()))
# cs.add_hyperparameter(CategoricalHyperparameter(
#     arch_prefix + f'prefix__multi', powerset(product(xfix_range, small_wrange)),
#     default_value=((1, 0), (2, 0), (3, 0))))
# cs.add_hyperparameter(CategoricalHyperparameter(
#     arch_prefix + f'suffix__multi', powerset(product(xfix_range, small_wrange)),
#     default_value=((1, 0), (2, 0), (3, 0))))

corenlp_prefix = 'corenlp_train_params__'
cs.add_hyperparameters([
    Constant(corenlp_prefix + 'sigmaSquared', .7676194187745077),
    Constant(corenlp_prefix + 'rareWordThresh', 6),
    Constant(corenlp_prefix + 'minFeatureThreshold', 1),
    Constant(corenlp_prefix + 'curWordMinFeatureThreshold', 4),
    Constant(corenlp_prefix + 'rareWordMinFeatureThresh', 1),
    Constant(corenlp_prefix + 'veryCommonWordThresh', 234),
])

cs.add_hyperparameters([
    CategoricalHyperparameter('augment_setvalued_targets', [False], default_value=False),
    CategoricalHyperparameter('filter_tags', (tuple(),), default_value=tuple()),
])

for i, rule in enumerate(
        [{'DDD', 'DDSA'}, {'AVG', 'PAVG'}, {'DGN', 'DGS'}, {'PAVD', 'PAVREL'}, {'AVG', 'AVW'}, {'DIN', 'DIS'},
         {'DRELS', 'PKOR'}, {'VAFIN.ind', 'VKFIN.ind'}, {'VAFIN.konj', 'VKFIN.konj'}, {'AVW', 'PAVG'},
         {'AVG', 'PAVREL'}, {'KON', 'KOUS'}, {'DDART', 'DDS'}, {'VKFIN.*', 'VKFIN.ind'}, {'DNEGS', 'PTKANT'}]):
    cs.add_hyperparameter(
        CategoricalHyperparameter(f'tag_expansion_rules__acc{i}', [np.NaN, tuple(rule)], default_value=np.NaN))

'''

'''
