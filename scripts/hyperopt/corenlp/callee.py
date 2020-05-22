import json
import re
from functools import reduce, partial
from itertools import groupby
from operator import add

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from scripts.hyperopt.corenlp.config_space_tagexp import cs
from setpos.data.split import is_masked, KFoldInDocSplitter
from setpos.tagger import CoreNLPTagger
from setpos.util import stopwatch


def parse_params(cfg):
    parsed_params = dict()

    params = pd.Series(dict(cfg)).dropna()
    # parent hyperparameter, only used for conditions in ConfigSpace. can't be used here
    # filter only applies to indices
    params = params.loc[~params.index.str.endswith('__enable')]
    # merge multikeys
    new_params = {}
    for k, g in groupby(params.sort_index().items(), key=lambda x: re.fullmatch(r'(.*?)(?:|__\d+|__acc\d+)', x[0])[1]):
        keys, vals = zip(*g)
        if len(vals) > 1:
            if re.fullmatch(r'(.*?)(?:__acc\d+)', keys[0]) is None:
                new_params[k + '__multi'] = vals
            else:
                new_params[k] = vals
        else:
            new_params[k] = vals[0]
    params = pd.Series(new_params)

    # simple top-level params
    singular_params = ~params.index.str.contains('__')
    for i, value in params[singular_params].items():
        parsed_params[i] = value
    params = params[~singular_params]

    # simple corenlp_train_params
    corenlp_train_params = dict()
    corenlp_simple = params.index.str.match('corenlp_train_params__(?!arch)')
    for i, value in params[corenlp_simple].items():
        corenlp_train_params['-' + re.match(r'corenlp_train_params__(.*)', i)[1]] = str(value)
    params = params[~corenlp_simple]

    # architecture params
    arch_params = []
    tag_offset = params.get('corenlp_train_params__arch__tag_window_offset', 0)
    params = params.drop('corenlp_train_params__arch__tag_window_offset', errors='ignore')
    corenlp_architecture_params = params.index.str.match('corenlp_train_params__arch.*')
    for i, value in params[corenlp_architecture_params].items():
        key = re.fullmatch(r'corenlp_train_params__arch__(.*?)(?:|__multi)', i)[1]
        if i.endswith('__multi'):
            values = [np.array(value_).flatten() for value_ in value]
        else:
            values = [np.array(value).flatten()]

        for value in values:
            if 'tag' in key.lower() or key == 'order':
                if key.lower().startswith('word'):
                    value[1:] += tag_offset
                else:
                    value[:] += tag_offset
            value = value.tolist()

        arch_params.append((key, value))

    if len(arch_params) > 15:
        raise ValueError('to many arguments in architecture')
    if arch_params:
        corenlp_train_params['-arch'] = ','.join(f'{f}({",".join(map(str, p))})' for f, p in arch_params)
    params = params[~corenlp_architecture_params]

    if not params.empty:
        # there should be nothing left
        print(params)
        raise ValueError()

    parsed_params['corenlp_train_params'] = list(reduce(add, corenlp_train_params.items()))
    return parsed_params


def evaluate(cfg, toks, tags, groups, timeout=1200, seed=0, k=5):
    params = parse_params(cfg)
    params_key = json.dumps(params)
    print(params_key)
    with stopwatch(verbose=False) as sw:
        if 'filter_tags' in params:
            mask = np.vectorize(partial(is_masked, prefixes=params['filter_tags']))(tags)
            toks, tags, groups = [a[mask] for a in [toks, tags, groups]]
        clf_class = params.get('clf', CoreNLPTagger)

        # remove higher level parameters
        params = {k: v for k, v in params.items() if k not in {'filter_tags', 'clf'}}

        clf = clf_class(**params, timeout=timeout, memlimit='6g')
        # Note: because of a SMAC bug, using loky in parallel will yield many error messages.
        # however they do not effect execution
        scores = cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(k=k, seed=seed), n_jobs=-1)

        elapsed = sw.elapsed
    score = scores.mean()
    print(dict(score=score, seed=seed, elapsed=elapsed, params=params_key, hash_of_param=hash(params_key)))

    return 1 - score


if __name__ == '__main__':
    print(parse_params(cs.sample_configuration()))
