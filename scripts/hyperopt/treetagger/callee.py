import json
from functools import reduce, partial
from operator import add

import numpy as np
from sklearn.model_selection import cross_val_score

from scripts.hyperopt.treetagger.config_space import cs
from setpos.data.split import is_masked, KFoldInDocSplitter
from setpos.tagger import TreeTagger
from setpos.util import stopwatch


def parse_params(cfg):
    parsed_params = dict()

    train_params = {f'-{k.split(":")[1]}': str(v) for k, v in dict(cfg).items() if k.startswith('train')}
    eval_params = {f'-{k.split(":")[1]}': str(v) for k, v in dict(cfg).items() if k.startswith('eval')}

    parsed_params['train_params'] = list(reduce(add, train_params.items()))
    parsed_params['eval_params'] = list(reduce(add, eval_params.items()))
    return parsed_params


def evaluate(cfg, toks, tags, groups, timeout=1200, seed=0, k=5):
    params = parse_params(cfg)
    params_key = json.dumps(params)
    print(params_key)
    with stopwatch(verbose=False) as sw:
        if 'filter_tags' in params:
            mask = np.vectorize(partial(is_masked, prefixes=params['filter_tags']))(tags)
            toks, tags, groups = [a[mask] for a in [toks, tags, groups]]
        clf_class = params.get('clf', TreeTagger)

        # remove higher level parameters
        params = {k: v for k, v in params.items() if k not in {'filter_tags', 'clf'}}

        clf = clf_class(**params)
        # Note: because of a SMAC bug, using loky in parallel will yield many error messages.
        # however they do not effect execution
        scores = cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(k=k, seed=seed), n_jobs=-1)

        elapsed = sw.elapsed
    score = scores.mean()
    print(dict(score=score, seed=seed, elapsed=elapsed, params=params_key, hash_of_param=hash(params_key)))

    return 1 - score


if __name__ == '__main__':
    print(parse_params(cs.sample_configuration()))
