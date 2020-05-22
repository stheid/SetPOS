from itertools import *

import numpy as np
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Hyperparameter, UniformIntegerHyperparameter


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

w_low, w_up = -3, 3
w_default_range = range(w_low, w_up + 1)
wchoices = [(i, j) for i in w_default_range for j in range(i + 1, w_up + 1)]
xfix_range = range(1, 4)
t_limit = 3
tchoices = [(i, j) for i in range(t_limit + 1) for j in range(i + 1, t_limit + 1)]
arch_prefix = 'corenlp_train_params__arch__'

cs.add_hyperparameter(CategoricalHyperparameter(arch_prefix + 'words', [(-2, 2), (-1, 1), (-3, 2)], (-1, 1)))
# cs.add_optional_hyperparameter(CategoricalHyperparameter(arch_prefix + 'biwords', wchoices, (-1, 1)))
# cs.add_optional_hyperparameter(CategoricalHyperparameter(arch_prefix + 'lowercasewords', wchoices, (-1, 1)))

cs.add_hyperparameter(UniformIntegerHyperparameter(arch_prefix + 'tag_window_offset', -2, -1, default_value=-2))
# cs.add_optional_hyperparameter(CategoricalHyperparameter(arch_prefix + 'tags', tchoices, (0, 2)))
# cs.add_optional_hyperparameter(CategoricalHyperparameter(arch_prefix + 'twoTags', tchoices, (0, 2)))
# cs.add_optional_hyperparameter(CategoricalHyperparameter(arch_prefix + 'treeTags', tchoices, (0, 2)))
cs.add_hyperparameter(CategoricalHyperparameter(arch_prefix + 'order', [(0, 2)], (0, 2)))
cs.add_hyperparameter(CategoricalHyperparameter(arch_prefix + f'wordTag__{0}', [np.NaN, (0, 0)], default_value=np.NaN))
cs.add_hyperparameter(CategoricalHyperparameter(arch_prefix + f'wordTag__{1}', [np.NaN, (0, 1)], default_value=np.NaN))
cs.add_hyperparameter(CategoricalHyperparameter(arch_prefix + f'wordTag__{2}', [np.NaN, (0, 2)], default_value=np.NaN))

small_wrange = range(w_low + 1, w_up)
# cs.add_hyperparameter(CategoricalHyperparameter(
#     arch_prefix + f'prefixsuffix__multi', powerset(xfix_range), default_value=()))
cs.add_hyperparameter(CategoricalHyperparameter(
    arch_prefix + f'prefix__multi', [(), ((1, 0), (2, 0), (3, 0))],
    default_value=()))
cs.add_hyperparameter(CategoricalHyperparameter(
    arch_prefix + f'suffix__multi', [(), ((2, 0),), ((1, 0), (2, 0), (3, 0)), ((2, 0), (3, 0), (4, 0), (5, 0))],
    default_value=()))

# corenlp_prefix = 'corenlp_train_params__'
# cs.add_hyperparameters([
#     UniformFloatHyperparameter(corenlp_prefix + 'sigmaSquared', .01, 10, default_value=.5, log=True),
#     UniformIntegerHyperparameter(corenlp_prefix + 'rareWordThresh', 0, 100, default_value=5),
#     UniformIntegerHyperparameter(corenlp_prefix + 'minFeatureThreshold', 0, 20, default_value=5),
#     UniformIntegerHyperparameter(corenlp_prefix + 'curWordMinFeatureThreshold', 1, 5, default_value=2),
#     UniformIntegerHyperparameter(corenlp_prefix + 'rareWordMinFeatureThresh', 1, 100, default_value=10, log=True),
#     UniformIntegerHyperparameter(corenlp_prefix + 'veryCommonWordThresh', 10, 500, default_value=250, log=True),
# ])

cs.add_hyperparameters([
    CategoricalHyperparameter('augment_setvalued_targets', [True, False], default_value=False),
    CategoricalHyperparameter('filter_tags', [(), ('$',)], default_value=('$',)),
])

# for i, rule in enumerate([{'DSG', 'DGN'}, {'DDSA', 'DDD'}, {'AVG', 'PAVG'}]):
#     cs.add_hyperparameter(
#         CategoricalHyperparameter(f'tag_expansion_rules__{i}', [np.NaN, tuple(rule)], default_value=np.NaN))

'''
INFO:smac.intensification.intensification.Intensifier:Updated estimated cost of incumbent on 1 runs: 0.1590    
INFO:smac.stats.stats.Stats:##########################################################
INFO:smac.stats.stats.Stats:Statistics:
INFO:smac.stats.stats.Stats:#Incumbent changed: 7
INFO:smac.stats.stats.Stats:#Target algorithm runs: 100 / 100.0                                                                                                                                  
INFO:smac.stats.stats.Stats:#Configurations: 100 
INFO:smac.stats.stats.Stats:Used wallclock time: 30954.58 / inf sec 
INFO:smac.stats.stats.Stats:Used target algorithm runtime: 30927.94 / 50400.00 sec
INFO:smac.stats.stats.Stats:##########################################################                                                                                                           
INFO:smac.facade.smac_hpo_facade.SMAC4HPO:Final Incumbent: Configuration:
  augment_setvalued_targets, Value: False
  corenlp_train_params__arch__order, Value: (0, 2)
  corenlp_train_params__arch__prefix__multi, Value: ((1, 0), (2, 0), (3, 0))                                                                                                                     
  corenlp_train_params__arch__suffix__multi, Value: ((2, 0), (3, 0), (4, 0), (5, 0))
  corenlp_train_params__arch__tag_window_offset, Value: -2
  corenlp_train_params__arch__wordTag__0, Value: nan
  corenlp_train_params__arch__wordTag__1, Value: (0, 1)                                                                                                                                          
  corenlp_train_params__arch__wordTag__2, Value: nan
  corenlp_train_params__arch__words, Value: (-3, 2)
  filter_tags, Value: ()          
{"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wordTag(0,-1),words(-3,2)"]}
'''