import re
from itertools import chain, groupby
from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.evaluation.general import SEED, clfs, evaluate, latexify, to_sc, outdir
from setpos.data.split import load, MCInDocSplitter
from setpos.tagger import MostFrequentTag, TreeTagger
from setpos.util import stopwatch


def calc_results(betas=None):
    toks, tags, groups = load(tag_prefix_masks=[])

    # train - test split
    train, test = next(MCInDocSplitter(seed=SEED).split(toks, tags, groups))

    # take the training data for train/eval cross-validation
    Xtrain, ytrain, _ = [l[train] for l in [toks, tags, groups]]
    Xeval, yeval, eval_g = [l[test] for l in [toks, tags, groups]]

    d = {}
    print(Xtrain.shape[0], Xeval.shape[0])

    for clf in clfs:
        clfname = 'baseline' if clf.get('clf', None) == MostFrequentTag else (
            'TreeTagger' if clf.get('clf', None) == TreeTagger else r'\textsc{c}ore\textsc{nlp}')
        # two clfs, 9 docs, 3 metrics -> two grouped barplots (over datasets), one for acc; util
        #           table (clf row, dataset col, subtable in cell for each metric)
        with stopwatch():
            old_clf = None
            for g, _ in groupby(eval_g):
                Xeval_, yeval_ = [l[eval_g == g] for l in [Xeval, yeval]]
                maskedtoks, maskedtags = np.vstack((Xtrain, Xeval_)), np.append(ytrain, yeval_)
                istrain = np.append(np.full(ytrain.shape, True), np.full(yeval_.shape, False))
                old_clf, scores = evaluate(clf, maskedtoks, maskedtags, istrain, oldclf=old_clf, betas=betas)

                d[(clfname, g)] = scores
        print(pd.DataFrame(d).T.loc[clfname].mean().tolist())
    print()
    return d


if __name__ == '__main__':
    savefile = 'in-doc.pkl'
    betas = [.2, .5, 1, 2, 5]
    if isfile(savefile):
        df = pd.read_pickle(savefile)
        new_idx = [(r'ml-' + s[3:] if s.startswith('\\') else s) for s in df.index]
        df.index = new_idx
        meta = pd.read_csv('../corpus_metadata/corpus.metadata.modified.csv')
        tokens = df.T.reset_index(level=1).level_1.apply(
            lambda x: meta[meta.sigle == x].Tokencount.iloc[0]).tolist()
        df = df.T.assign(tokens=tokens).T
    else:
        d = calc_results(betas=betas)
        df = pd.DataFrame(d, index=['ml-acc'] + list(chain.from_iterable(
            [[f'ml-util $\\beta={beta}$', f'$\\beta={beta}$'] for beta in betas])))
        df.to_pickle(savefile)

    latexify(columns=2)

    losses = [idx for idx in df.index if 'ml' in idx]
    f, (ax, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw=dict(width_ratios=[1, 5], wspace=0.05))

    df.loc[['ml-acc']].rename({'ml-acc': 'acc'}).T.mean(level=0).T \
        .rename(to_sc, axis='columns').plot.bar(ax=ax, yerr=df.loc[['ml-acc']].rename({'ml-acc': 'acc'}).T.std(
        level=0).T.rename(to_sc, axis='columns'), legend=False)
    df.loc[losses[1:]].rename(dict(zip(losses[1:], [f'${beta}$' for beta in betas]))).T.mean(level=0).T \
        .rename(to_sc, axis='columns').plot.bar(ax=ax2, yerr=df.loc[losses[1:]].rename(
        dict(zip(losses[1:], [f'${beta}$' for beta in betas]))).T.std(level=0).T.rename(to_sc, axis='columns'))
    ax.tick_params(axis='x', rotation=0)
    ax2.tick_params(axis='x', rotation=0)
    ax2.set_xlabel('util / $\\beta$')
    ax.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax2.yaxis.tick_right()
    ax.grid(which='both', axis='y')
    ax2.grid(which='both', axis='y')
    ax2.legend(loc='lower right')
    ax.set_ylim(bottom=.5, top=1)
    ax2.set_ylim(bottom=.5, top=1)
    f.suptitle('Predictive Performance', fontsize=8, y=.962)
    f.savefig(outdir + 'in-doc-acc-normalmean.pdf', bbox_inches='tight')

    # ax = df.loc[losses+['tokens']].T.groupby(level=0,sort=False).apply(wavg,losses,'tokens').T.plot.bar(yerr=df.loc[losses].T.std(level=0).T)
    # ax.tick_params(axis='x', rotation=0)
    # ax.set_axisbelow(True)
    # ax.grid(which='both', axis='y')
    # ax.legend(loc='lower right')
    # ax.set_ylim(bottom=.5, top=1)
    # ax.set_title('Predictive Performance')
    # fig = ax.get_figure()
    # fig.savefig(outdir + 'in-doc-acc-weigthedmean.pdf', bbox_inches='tight')
    #
    #
    # df2 = df.loc[losses+['tokens']].T.groupby(level=0,sort=False).apply(wavg,losses,'tokens').T - df.loc[losses].T.mean(level=0).T
    # ax = df2.plot.bar()
    # ax.tick_params(axis='x', rotation=0)
    # ax.set_axisbelow(True)
    # ax.grid(which='both', axis='y')
    # ax.legend(loc='lower right')
    # ax.set_title('Predictive Performance')
    # fig = ax.get_figure()
    # fig.savefig(outdir + 'in-doc-acc-wavg-avg_diff.pdf', bbox_inches='tight')

    setsize = [idx for idx in df.index if 'ml-' not in idx and idx != 'tokens']
    ax = df.loc[setsize].T.mean(level=0).T \
        .rename(to_sc, axis='columns').plot.bar(yerr=df.loc[setsize].T.std(level=0).T.rename(to_sc, axis='columns'))
    ax.tick_params(axis='x', rotation=0)
    ax.legend(loc='upper left')
    ax.set_title('Set Size')
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y')
    ax.set_ylim(bottom=0)
    fig = ax.get_figure()
    fig.savefig(outdir + 'in-doc-setsize.pdf', bbox_inches='tight')

    for loss in losses:
        df.loc[loss] = df.loc[loss].apply(
            lambda x: re.sub(r'(\d+)\.(\d+)', r'\\llap{\1}.\\rlap{\2}', '{:0.2%}'.format(x).replace('%', '')))
    for size in setsize:
        df.loc[size] = df.loc[size].apply(
            lambda x: re.sub(r'(\d+)\.(\d+)', r'\\llap{\1}.\\rlap{\2}', '{:0.2f}'.format(x)))
    # with open(outdir + 'in-doc.tex', 'w') as f:
    # &              &       \multicolumn{2}{c}{$ \beta=0.2 $}
    # &       \multicolumn{2}{c}{$ \beta=0.5 $}
    # &       \multicolumn{2}{c}{$ \beta=1 $}
    # &       \multicolumn{2}{c}{$ \beta=2 $}
    # &       \multicolumn{2}{c}{$ \beta=5 $} \\
    #  ml-acc &  ml-util  & $|\hat{Y}|$ &  ml-util  & $|\hat{Y}|$ &  ml-util  &
    #  $|\hat{Y}|$ &ml-util  & $|\hat{Y}|$ & ml-util  & $|\hat{Y}|$ \\
    #   df.T.to_latex(buf=f, escape=False, multicolumn_format='c', multirow=True,
    #                column_format='l' + '@{\enspace}c' * 11,
    #               )
