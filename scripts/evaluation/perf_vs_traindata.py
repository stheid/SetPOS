import re
from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.evaluation.general import SEED, clfs, evaluate, latexify
from setpos.data.split import load, MCInDocSplitter
from setpos.tagger import MostFrequentTag
from setpos.util import stopwatch


def calc_results():
    doc = 'St2'
    parts = 50
    toks, tags, groups = load(tag_prefix_masks=[])

    # train - test split
    train, test = next(MCInDocSplitter(seed=SEED).split(toks, tags, groups))

    # take the training data for train/eval cross-validation
    x_t, y_t, g_t = [l[train] for l in [toks, tags, groups]]
    x_e, y_e, g_e = [l[test] for l in [toks, tags, groups]]
    x_t, y_t = [l[g_t == doc] for l in [x_t, y_t]]
    x_e, y_e = [l[g_e == doc] for l in [x_e, y_e]]

    d = {}
    for frac in (1 + np.array([-.5, 0, 1, 2, 5, 10, 19, 34, 49])) * x_t.shape[0] / parts:
        frac = int(frac)
        print(frac, x_e.shape[0])
        x_t_, y_t_ = [l[:frac] for l in [x_t, y_t]]
        maskedtoks, maskedtags = np.vstack((x_t_, x_e)), np.append(y_t_, y_e)
        for clf in clfs:
            istrain = np.append(np.full(y_t_.shape, True), np.full(y_e.shape, False))
            clfname = 'baseline' if clf.get('clf', None) == MostFrequentTag else r'\textsc{c}ore\textsc{nlp}'
            # two clfs, 10 fractions, 1 doc, 1 metrics -> lineplots (both clfs) (over fractions) with marks and dashed line
            #           table (clf row, fracs col)
            with stopwatch():
                _, scores = evaluate(clf, maskedtoks, maskedtags, istrain)
            d[(clfname, frac)] = scores
            print(scores)
        print()
    return d


if __name__ == '__main__':
    outdir = '../../../../Thesis/eval/'
    savefile = 'perf_vs_traindata.pkl'
    if isfile(savefile):
        df = pd.read_pickle(savefile)
        df.columns = 'ml-acc,ml-util,set size'.split(',')
    else:
        d = calc_results()
        df = pd.DataFrame(d).T
        df.columns = 'ml-acc,ml-util,set size'.split(',')
        df.to_pickle(savefile)

    latexify(height_mul=.5)
    df2 = df[['ml-acc', 'ml-util']].unstack(0)
    df2 = df2.stack(0).stack(0).reset_index()
    df2.columns = ['Token count', 'Measure', 'Clf', 'Score']
    df2 = df2.sort_values('Clf', ascending=False)
    ax = sns.lineplot(x='Token count', y='Score', hue='Clf', style='Measure', data=df2, markers=True)
    ax.set_ylim(top=1, bottom=.4)
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y')
    plt.tight_layout()
    ax.get_figure().savefig(outdir + savefile[:-4] + '_scores.pdf')
    plt.clf()

    df2 = df[['set size']].unstack(0)
    df2 = df2.stack(0).stack(0).reset_index()
    df2.columns = ['Token count', 'Measure', 'Clf', 'Set size']
    ax = sns.lineplot(x='Token count', y='Set size', hue='Clf', data=df2, marker='o')
    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y')
    plt.tight_layout()
    ax.get_figure().savefig(outdir + savefile[:-4] + '_setsize.pdf')

    df['ml-acc'] = df['ml-acc'].apply(
        lambda x: re.sub(r'(\d+)\.(\d+)', r'\\llap{\1}.\\rlap{\2}', '{:0.2%}'.format(x).replace('%', '')))
    df['ml-util'] = df['ml-util'].apply(
        lambda x: re.sub(r'(\d+)\.(\d+)', r'\\llap{\1}.\\rlap{\2}', '{:0.2%}'.format(x).replace('%', '')))
    df['set size'] = df['set size'].apply(
        lambda x: re.sub(r'(\d+)\.(\d+)', r'\\llap{\1}.\\rlap{\2}', '{:0.2f}'.format(x)))
    df.columns = '\mlacc,\mlutil,$|\hat{Y}|$'.split(',')
    with open(outdir + savefile[:-3] + 'tex', 'w') as f:
        df = df.T.stack(0)
        df.to_latex(buf=f, escape=False, multicolumn_format='c', multirow=True,
                    column_format='l@{ }l' + '@{\enspace}c' * 9,
                    header=[r'\makebox[\widthof{13509}]{' + str(col) + '}' for col in df.columns])
