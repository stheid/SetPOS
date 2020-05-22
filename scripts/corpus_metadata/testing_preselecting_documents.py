import os
import re
from contextlib import redirect_stdout
from glob import glob
from itertools import chain, groupby
from os.path import basename, isfile
from random import random

import pandas as pd
from joblib import Parallel, delayed
from numpy.linalg import norm

from setpos.data.split import load
from setpos.tagger import CoreNLPTagger


def memoize(func):
    cache = {}

    # parse old results and fill cache with it
    for file in glob('out/*.csv'):
        prefix = re.match('^([a-zA-Z0-9]+)-', basename(file))[1]
        df = pd.read_csv(file)
        for i, row in df.iterrows():
            cache[(prefix, frozenset(eval(row.train_docs)))] = dict(total=row.total, known=row.known, unk=row.unk)

    def decorator(train, eval):
        key = (eval, frozenset(train))
        if key not in cache:
            res = func(train, eval)
            cache[key] = res
        else:
            print('!!!cachehit!!!')
            res = cache[key]
        return res

    return decorator


@memoize
def evaluate(train, eval):
    with redirect_stdout(None):
        clf = CoreNLPTagger()
        mask = pd.Series(groups).apply(lambda x: x in train).to_numpy()
        clf.fit(X[mask], y[mask])
        print("fitted")
        _, result_df = clf.score(X[groups == eval], y[groups == eval], long_result=True)
    return result_df.loc['accuracy'].to_dict()


def dist(a: pd.Series, b: pd.Series, year_weight=1):
    absolute_difference = a.to_numpy() - b.to_numpy()
    year_index = list(a.index).index("year_norm")
    absolute_difference[year_index] *= year_weight
    return norm(absolute_difference)


def distance_sorted_list(selector, metadata, year_weight=1):
    dists = [(dist(metadata.loc[selector], row, year_weight), i) for i, row in metadata.iterrows()]
    dists = [(k, [v[1] for v in g]) for k, g in groupby(sorted(dists), key=lambda x: round(x[0], 6))]
    # print(f"max dist: {dists[-1][0]:.2f}")
    return dists


def print_and_eval(train_docs, eval_doc, end_index, dists, ):
    dist = dists[end_index][0]
    acc = evaluate(train_docs, eval_doc)

    print("curr max dist:", dist)
    print("newly added:", dists[end_index][1])
    print("training docs:", train_docs)
    print("excluded docs:", set(doc_titles - train_docs))
    for i, val in acc.items():
        print(f'{i}: {val:.4f}')
    print()
    return dict(dist=dist, train_docs=train_docs, **acc)


if __name__ == '__main__':
    X, y, groups = load()
    doc_titles = set(groups)

    metadata = pd.read_csv("corpus.metadata.csv", index_col='sigle',
                           usecols=["sigle", 'year_norm', 'lon_norm', 'lat_norm'])
    os.makedirs("out", exist_ok=True)

    for seed in range(4):
        for selector in doc_titles:
            prefix = f'{selector}-seed:{seed}'
            if isfile(prefix + '.csv'):
                continue
            print('test doc:', prefix, end="\n-------------------------\n\n")

            # calculate sorting of potential training docs to add incrementally
            # dists = distance_sorted_list(selector, metadata, year_weight)
            dists = [(i, [v]) for i, v in enumerate(sorted(set(metadata.index) - {selector}, key=lambda x: random()))]

            # create train_sets for evaluation
            train_sets_and_cutoff = []
            for end_index in range(len(dists)):
                train_docs = chain.from_iterable([docs for dist, docs in dists[:end_index + 1]])
                train_docs = set(train_docs) - {selector}
                if not train_docs:
                    # this will happen if the only document with 0 distance is the training document itself.
                    continue
                train_sets_and_cutoff.append([train_docs, end_index])

            results_df = pd.DataFrame(
                Parallel(1)(
                    delayed(print_and_eval)(train, selector, cutoff, dists) for train, cutoff in train_sets_and_cutoff)) \
                .set_index('dist')
            results_df[['unk', 'known', 'total']].plot(title=selector).get_figure().savefig(
                f"out/{prefix}.png")
            results_df.to_csv(f'out/{prefix}.csv')
