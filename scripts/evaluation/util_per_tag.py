import json
import operator
import re
from collections import Counter
from functools import reduce, partial
from itertools import chain
from operator import itemgetter
from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.evaluation.general import SEED, evaluate, latexify, clf
from setpos.data.split import load, MCInDocSplitter
from setpos.util import stopwatch


def calc_results():
    toks, tags, groups = load(tag_prefix_masks=[])

    # train - test split
    train, test = next(MCInDocSplitter(seed=SEED).split(toks, tags, groups))

    # take the training data for train/eval cross-validation
    Xtrain, ytrain, _ = [l[train] for l in [toks, tags, groups]]
    Xeval, yeval, _ = [l[test] for l in [toks, tags, groups]]

    maskedtoks, maskedtags = np.vstack((Xtrain, Xeval)), np.append(ytrain, yeval)
    istrain = np.append(np.full(ytrain.shape, True), np.full(yeval.shape, False))
    with stopwatch():
        df, _ = evaluate(clf, maskedtoks, maskedtags, istrain, raw=True)
    return df


def get_ytrain():
    toks, tags, groups = load(tag_prefix_masks=[])

    # train - test split
    train, _ = next(MCInDocSplitter(seed=SEED).split(toks, tags, groups))

    # take the training data for train/eval cross-validation
    _, ytrain, _ = [l[train] for l in [toks, tags, groups]]
    return ytrain


if __name__ == '__main__':
    outdir = '../../../../Thesis/eval/'
    savefile = 'util-per-tag.pkl'
    if isfile(savefile):
        df = pd.read_pickle(savefile)
    else:
        df = calc_results()
        df.to_pickle(savefile)

    tags = df.target.apply(set).aggregate(partial(reduce, operator.or_))
    tags = {tag: df.target[df.target.apply(lambda x: tag in x)].count() for tag in tags}

    min_count = 20
    filtered_tags = {k: v for k, v in tags.items() if v > min_count}
    accs = {tag: df.iscorrect[df.target.apply(lambda x: tag in x)].mean() for tag, count in tags.items() if
            count >= min_count}
    util = {tag: df.const_util[df.target.apply(lambda x: tag in x)].mean() for tag, count in tags.items() if
            count >= min_count}
    # count = {tag: df.const_util[df.target.apply(lambda x: tag in x)].count() for tag, count in tags.items() if
    #         count >= min_count}
    ytrain = get_ytrain()
    count = Counter([t for t in chain.from_iterable(pd.Series(ytrain).apply(lambda x: list(json.loads(x).keys()))) if
                     t in filtered_tags.keys()])
    tag = 'DDA'
    print(count[tag], end=' ')
    count = pd.Series(count) / pd.Series(count).sum()
    print(count[tag], accs[tag], util[tag])
    count = count / pd.Series(count).max()

    df2 = pd.DataFrame([pd.Series(accs), count], index=['ml-acc', 'relative frequency']).T.sort_index()
    print(df2.corr())
    df2 = df2.stack().unstack(0).unstack(0).reset_index()
    df2.columns = ['Tag', 'Type', 'Value']

    latexify(fig_width=6.2)
    ax = sns.barplot(x='Tag', y='Value', hue='Type', data=df2)
    ax.tick_params(axis='x', rotation=90)
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y')
    ax.legend(loc=(.712, .86))
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.tight_layout()
    ax.get_figure().savefig(outdir + savefile[:-3] + 'pgf')

    # error analysis table
    errors = {tag: df.iscorrect[df.target.apply(lambda x: tag in x)].apply(lambda x: not bool(x)).sum() for tag, count
              in tags.items()}
    appr_pavap_convs = df.pred[df.target.apply(lambda x_: 'APPR' in x_)].apply(lambda x_: x_ == 'PAVAP').sum() \
                       + df.pred[df.target.apply(lambda x_: 'PAVAP' in x_)].apply(lambda x_: x_ == 'APPR').sum()
    print('total confusions of APPR and PAVAP:', appr_pavap_convs)
    errors = sorted(errors.items(), key=itemgetter(1), reverse=True)

    percent_to_str = lambda x: re.sub(r'(\d+)\.(.*)', r'\\llap{\1}.\\rlap{\2}', f'{x:.2%}'.replace('%', '\%'))
    convert_sc = lambda x: re.sub(r'([A-Z]+)', lambda x: f'\\textsc{{{x.group().lower()}}}', x)

    total_errs = sum(x[1] for x in errors)
    explained_errs = 0
    for x in errors[:10]:
        acc = {tag: df.pred[df.target.apply(lambda x_: x[0] in x_)].apply(lambda x_: x_ == tag).sum() for tag in tags}
        del acc[x[0]]
        err = sorted(acc.items(), key=itemgetter(1), reverse=True)
        i = 0
        while sum(map(itemgetter(1), err[:i])) < x[1] * .75:
            i += 1
        i = min(5, i)
        err_s = ', '.join([convert_sc(e[0]) + '\,(' + str(e[1]) + ')' for e in err[:i]])
        total_part = sum(map(itemgetter(1), err[:i])) / total_errs

        print(convert_sc(x[0]),  # percent_to_str(total_part),
              err_s,
              f'{x[1]:,}',
              percent_to_str(accs[x[0]]),
              sep='&', end='\\\\\n')
        explained_errs += sum(map(itemgetter(1), err[:i]))

    print(f'percent of errors explained: {explained_errs / total_errs:.2%}')
