import shutil
from os import path
from os.path import isfile

import numpy as np
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger

from scripts.hyperopt.MySobol import SobolDesign
from scripts.hyperopt.treetagger import callee
from scripts.hyperopt.treetagger.config_space import cs
from setpos.data.split import MCInDocSplitter, load


def create_or_restore_smac(scenario_dict, rng, tae):
    out_dir = path.join(scenario_dict['output_dir'], 'run_1')
    if True or not isfile(path.join(out_dir, "traj_aclib2.json")):
        # if some incomplete data lays arround, delete it completely
        shutil.rmtree(out_dir, ignore_errors=True)
        scenario = Scenario(scenario_dict)
        smac = SMAC4HPO(scenario=scenario, rng=rng, tae_runner=tae, initial_design=SobolDesign, run_id=1)
    else:
        new_scenario = Scenario(scenario_dict)
        rh_path = path.join(out_dir, "runhistory.json")
        runhistory = RunHistory(aggregate_func=None)
        runhistory.load_json(rh_path, new_scenario.cs)
        # ... stats, ...
        stats_path = path.join(out_dir, "stats.json")
        stats = Stats(new_scenario)
        stats.load(stats_path)
        # ... and trajectory.
        traj_path = path.join(out_dir, "traj_aclib2.json")
        trajectory = TrajLogger.read_traj_aclib_format(
            fn=traj_path, cs=new_scenario.cs)
        incumbent = trajectory[-1]["incumbent"]

        # Now we can initialize SMAC with the recovered objects and restore the
        # state where we left off. By providing stats and a restore_incumbent, SMAC
        # automatically detects the intention of restoring a state.
        smac = SMAC4HPO(scenario=new_scenario,
                        runhistory=runhistory,
                        stats=stats,
                        restore_incumbent=incumbent,
                        run_id=1)
        print('restored smac from:', out_dir)
    return smac


def optimize():
    SEED = 1

    toks, tags, groups = load(tag_prefix_masks=[])  # [l[:3000] for l in load(tag_prefix_masks=[])]  #
    # train - test split
    train, _ = next(MCInDocSplitter(seed=SEED).split(toks, tags, groups))

    # take the training data for train/eval cross-validation
    toks, tags, groups = [l[train] for l in [toks, tags, groups]]

    def tae(cfg, seed=0):
        return callee.evaluate(cfg, toks, tags, groups, timeout=1200, seed=seed, k=5)

    scenario_dict = {'run_obj': 'quality',  # we optimize quality (alternatively runtime)
                     'runcount-limit': 100,
                     'algo_runs_timelimit': 60 * 60 * 14,
                     'cutoff': 1200,  # stop algorithms after 10x default runtime
                     "cs": cs,  # configuration space
                     "deterministic": 'true',
                     'output_dir': 'smac3_test_treetagger'
                     }

    smac = create_or_restore_smac(scenario_dict=scenario_dict, rng=np.random.RandomState(SEED), tae=tae)
    incumbent = smac.optimize()

    inc_value = tae(incumbent)

    print("Optimized Value: %.2f" % (inc_value))


if __name__ == '__main__':
    optimize()
