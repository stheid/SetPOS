import re
from collections import Counter
from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from scripts.evaluation.general import SEED, clfs, evaluate, latexify, to_sc, outdir
from setpos.data.split import load, MCInDocSplitter
from setpos.tagger import MostFrequentTag, TreeTagger


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
        mask = groups_t == doc
        mask_e = groups_e == doc
        print(doc, np.sum(mask), np.sum(mask_e))
        Xtrain, ytrain = toks_t[mask], tags_t[mask]
        Xeval, yeval = toks_e[mask_e], tags_e[mask_e]
        maskedtoks, maskedtags = np.vstack((Xtrain, Xeval)), np.append(ytrain, yeval)
        istrain = np.append(np.full(ytrain.shape, True), np.full(yeval.shape, False))
        for clf in clfs:
            clfname = 'baseline' if clf.get('clf', None) == MostFrequentTag else (
                'TreeTagger' if clf.get('clf', None) == TreeTagger else r'\textsc{c}ore\textsc{nlp}')
            doc = ''.join(((r'\textsc{' + c.lower() + '}' if re.match('[A-Z]', c) != None else c) for c in doc))
            # two classifiers, k datasets, 3 metrics -> two grouped barplots (over datasets), one for acc;util
            #           table (clf row, dataset col, subtable in cell for each metric)
            _, scores = evaluate(clf, maskedtoks, maskedtags, istrain)
            d[(doc, clfname)] = scores
            print(scores)
    print()
    return d


if __name__ == '__main__':
    savefile = 'in-domain.pkl'
    if isfile(savefile):
        df = pd.read_pickle(savefile)
    else:
        d = calc_results()
        df = pd.DataFrame(d, index=('\mlacc', '\mlutil', '$|\hat{Y}|$'))
        df.to_pickle(savefile)

    latexify(columns=1, width_mul=1.08, height_mul=.40)
    f = plt.figure()
    gs0 = gridspec.GridSpec(1, 2, figure=f, width_ratios=[2, 1], wspace=.1)
    gs00 = gs0[0].subgridspec(1, 2, wspace=.1)
    w = .66

    ax1 = f.add_subplot(gs00[0])
    ax2 = f.add_subplot(gs00[1])
    ax3 = f.add_subplot(gs0[1])

    df = df.stack(1)
    df = df[[r'\textsc{d}\textsc{s}\textsc{r}', r'\textsc{r}\textsc{e}\textsc{n}4', r'\textsc{r}\textsc{e}\textsc{n}14',
             r'\textsc{k}o']]
    df.columns = ['Duisburg', 'Bremen', 'Bamberg', 'Kołobrzeg']
    df = df.unstack(-1)

    ax = ax1
    df.loc['\mlacc'].unstack(-1)[['baseline', 'TreeTagger', r'\textsc{c}ore\textsc{nlp}']] \
        .rename(to_sc, axis='columns').plot.bar(ax=ax, legend=False, width=w)
    ax.tick_params(axis='x', rotation=20)
    ax.set_ylim(bottom=.4, top=1)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end + .1, .1))
    ax.set_title('Accuracy')
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y')

    ax = ax2
    df.loc['\mlutil'].unstack(-1)[['baseline', 'TreeTagger', r'\textsc{c}ore\textsc{nlp}']] \
        .rename(to_sc, axis='columns').plot.bar(ax=ax, width=w)
    ax.tick_params(axis='x', rotation=20)
    ax.yaxis.tick_right()
    ax.set_ylim(bottom=.4, top=1)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end + .1, .1))
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y')
    ax.set_title('Utility')
    ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=False, labelright=False)
    ax.legend(loc='lower right')

    ax = ax3
    df.loc['$|\hat{Y}|$'].unstack(-1)[['baseline', 'TreeTagger', r'\textsc{c}ore\textsc{nlp}']] \
        .rename(to_sc, axis='columns').plot.bar(ax=ax, legend=False, width=w)
    ax.tick_params(axis='x', rotation=20)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end + 1, 1))
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y')
    ax.set_title('Set Size')
    f.savefig(outdir + 'in-domain_all.pdf', bbox_inches='tight')

    df.loc['\mlacc'] = df.loc['\mlacc'].apply(
        lambda x: re.sub(r'(\d+)\.(\d+)', r'\\llap{\1}.\\rlap{\2}', '{:0.2%}'.format(x).replace('%', '')))
    df.loc['\mlutil'] = df.loc['\mlutil'].apply(
        lambda x: re.sub(r'(\d+)\.(\d+)', r'\\llap{\1}.\\rlap{\2}', '{:0.2%}'.format(x).replace('%', '')))
    df.loc['$|\hat{Y}|$'] = df.loc['$|\hat{Y}|$'].apply(
        lambda x: re.sub(r'(\d+)\.(\d+)', r'\\llap{\1}.\\rlap{\2}', '{:0.2f}'.format(x)))
    # with open(outdir + 'in-domain.tex', 'w') as f:
    #    df = df.stack(-1)
    #    df.to_latex(buf=f, escape=False, multicolumn_format='c', multirow=True,
    #                column_format='l@{ }l' + '@{\enspace}c' * 9,
    #                header=[r'\makebox[\widthof{\textsc{ren14}}]{' + col + '}' for col in df.columns])

    # plt.clf()
    # latexify()
    # meta = pd.read_csv('../../scripts/issue47/corpus.metadata.csv')
    # meta = meta[meta.Tokencount > 7000].sort_values('sigle')
    # ax = sns.barplot(data=meta, x='sigle', y='Tokencount')
    # ax.set_xlabel('')
    # ax.get_figure().savefig(outdir + 'largedocs_prior.pgf', bbox_inches='tight')
