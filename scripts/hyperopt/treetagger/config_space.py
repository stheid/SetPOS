from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter

cs = ConfigurationSpace()

# train params
prefix = 'train:'
cs.add_hyperparameters([
    UniformIntegerHyperparameter(prefix + 'cl', 1, 4, default_value=2),
    UniformFloatHyperparameter(prefix + 'dtg', 0, 1, default_value=.5),
    UniformFloatHyperparameter(prefix + 'sw', 0, 5, default_value=1),
    UniformFloatHyperparameter(prefix + 'ecw', 0, 1, default_value=.15),
    UniformFloatHyperparameter(prefix + 'atg', 0, 5, default_value=1.2),
    UniformFloatHyperparameter(prefix + 'lt', 0, 1, default_value=0)
])

# eval params
prefix = 'eval:'
cs.add_hyperparameters([
    UniformFloatHyperparameter(prefix + 'eps', 0, 1, default_value=0.1),
    UniformFloatHyperparameter(prefix + 'beam', 0.00001, 0.001, default_value=0.0001, log=True)
])
