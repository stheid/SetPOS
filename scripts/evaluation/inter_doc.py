import re
from collections import Counter
from os.path import isfile

import numpy as np
import pandas as pd

from scripts.evaluation.general import SEED, clfs, evaluate, latexify, outdir
from setpos.data.split import load, MCInDocSplitter
from setpos.tagger import MostFrequentTag, TreeTagger
from setpos.util import stopwatch


def calc_results():
    toks, tags, groups = load(tag_prefix_masks=[])  # [l[:3000] for l in load(tag_prefix_masks=[])]  #
    ctr = Counter(groups)

    # train - test split
    train, test = next(MCInDocSplitter(seed=SEED).split(toks, tags, groups))

    # take the training data for train/eval cross-validation
    toks_t, tags_t, groups_t = [l[train] for l in [toks, tags, groups]]

    # take the training data for train/eval cross-validation
    toks_e, tags_e, groups_e = [l[test] for l in [toks, tags, groups]]

    d = {}
    for doc, _ in filter(lambda t: t[1] > 7000, ctr.items()):
        mask = groups != doc
        mask_e = groups_e == doc
        Xtrain, ytrain = toks[mask], tags[mask]
        Xeval, yeval = toks_e[mask_e], tags_e[mask_e]
        print(doc, np.sum(mask), np.sum(mask_e))
        maskedtoks, maskedtags = np.vstack((Xtrain, Xeval)), np.append(ytrain, yeval)
        for clf in clfs:
            istrain = np.append(np.full(ytrain.shape, True), np.full(yeval.shape, False))

            clfname = 'baseline' if clf.get('clf', None) == MostFrequentTag else (
                'TreeTagger' if clf.get('clf', None) == TreeTagger else r'\textsc{c}ore\textsc{nlp}')
            docname = ''.join(((r'\textsc{' + c.lower() + '}' if re.match('[A-Z]', c) != None else c) for c in doc))
            # two clfs, 9 docs, 3 metrics -> two grouped barplots (over datasets), one for acc; util
            #           table (clf row, dataset col, subtable in cell for each metric)
            with stopwatch():
                _, scores = evaluate(clf, maskedtoks, maskedtags, istrain)
            d[(docname, clfname)] = scores
            print(scores)
    print()
    return d


if __name__ == '__main__':
    savefile = 'inter-doc.pkl'
    if isfile(savefile):
        df = pd.read_pickle(savefile)
    else:
        d = calc_results()
        df = pd.DataFrame(d, index=('\mlacc', '\mlutil', '$|\hat{Y}|$'))
        df.to_pickle(savefile)

    latexify(height_mul=.5)

    ax = df.loc['\mlacc'].unstack(-1)[['baseline', 'TreeTagger', r'\textsc{c}ore\textsc{nlp}']].plot.bar()
    ax.tick_params(axis='x', rotation=0)
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y')
    ax.legend(loc='lower right')
    ax.set_ylim(bottom=.4, top=1)
    ax.set_title('mlacc')
    fig = ax.get_figure()
    fig.savefig(outdir + savefile[:-4] + '-acc.pdf', bbox_inches='tight')

    ax = df.loc['\mlutil'].unstack(-1)[['baseline', 'TreeTagger', r'\textsc{c}ore\textsc{nlp}']].plot.bar()
    ax.tick_params(axis='x', rotation=0)
    ax.legend(loc='lower right')
    ax.set_title('mlutil')
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y')
    ax.set_ylim(bottom=.4, top=1)
    fig = ax.get_figure()
    fig.savefig(outdir + savefile[:-4] + '-util.pdf', bbox_inches='tight')

    df.loc['\mlacc'] = df.loc['\mlacc'].apply(
        lambda x: re.sub(r'(\d+)\.(\d+)', r'\\llap{\1}.\\rlap{\2}', '{:0.2%}'.format(x).replace('%', '')))
    df.loc['\mlutil'] = df.loc['\mlutil'].apply(
        lambda x: re.sub(r'(\d+)\.(\d+)', r'\\llap{\1}.\\rlap{\2}', '{:0.2%}'.format(x).replace('%', '')))
    df.loc['$|\hat{Y}|$'] = df.loc['$|\hat{Y}|$'].apply(
        lambda x: re.sub(r'(\d+)\.(\d+)', r'\\llap{\1}.\\rlap{\2}', '{:0.2f}'.format(x)))
    with open(outdir + savefile[:-4] + '.tex', 'w') as f:
        df = df.stack(-1)
        df.to_latex(buf=f, escape=False, multicolumn_format='c', multirow=True,
                    column_format='l@{ }l' + '@{\enspace}c' * 9,
                    header=[r'\makebox[\widthof{\textsc{ren14}}]{' + col + '}' for col in df.columns])
