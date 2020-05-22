import re
from os.path import isfile

import numpy as np
import pandas as pd

from scripts.evaluation.general import SEED, clfs, evaluate
from setpos.data.split import load, MCInDocSplitter
from setpos.tagger import MostFrequentTag
from setpos.util import stopwatch


def calc_results():
    toks, tags, groups = load(tag_prefix_masks=[])  # [l[:3000] for l in load(tag_prefix_masks=[])]  #

    # train - test split
    train, test = next(MCInDocSplitter(seed=SEED).split(toks, tags, groups))

    # take the training data for train/eval cross-validation
    Xtrain, ytrain, _ = [l[train] for l in [toks, tags, groups]]
    Xeval, yeval, _ = [l[test] for l in [toks, tags, groups]]

    maskedtoks, maskedtags = np.vstack((Xtrain, Xeval)), np.append(ytrain, yeval)
    istrain = np.append(np.full(ytrain.shape, True), np.full(yeval.shape, False))

    d = {}
    for params in clfs:
        if params.get('clf', None) == MostFrequentTag:
            clfname = 'baseline'
            clf, total_score = evaluate(params, maskedtoks, maskedtags, istrain)
            print('known word frac:', clf.knownwords(Xeval))
            clf.scope = 'known'
            _, known_scores = evaluate(params, maskedtoks, maskedtags, istrain, oldclf=clf)
            clf.scope = 'unk'
            _, unk_scores = evaluate(params, maskedtoks, maskedtags, istrain, oldclf=clf)
            df = pd.DataFrame([unk_scores, known_scores, total_score]).T
        else:
            clfname = r'\textsc{c}ore\textsc{nlp}'
            with stopwatch():
                # 2clf, 3 scopes, 3 metrics
                _, score = evaluate(params, maskedtoks, maskedtags, istrain, raw=True)
            df = pd.DataFrame([score.loc[measure, :] for measure in ['accuracy', 'avg util', 'avg setsize']])
        df.index = ['Accuracy', 'Utility', 'Set size']
        df.columns = ['Unknown', 'Known', 'Total']
        for col in df:
            d[(clfname, col)] = df[col]
    print()
    return d


if __name__ == '__main__':
    outdir = '../../../../Thesis/eval/'
    savefile = 'unk-word-perf.pkl'
    if isfile(savefile):
        df = pd.read_pickle(savefile)
    else:
        df = pd.DataFrame(calc_results())
        df.to_pickle(savefile)

    df = pd.read_pickle('unk-word-perf.pkl')
    df.index = ['\mlacc', '\mlutil', '$|\hat{Y}|$']
    df.loc['\mlacc'] = df.loc['\mlacc'].apply(
        lambda x: re.sub(r'(\d+)\.(.*)', r'\\llap{\1}.\\rlap{\2}', '{:0.2%}'.format(x)).replace('%', ''))
    df.loc['\mlutil'] = df.loc['\mlutil'].apply(
        lambda x: re.sub(r'(\d+)\.(.*)', r'\\llap{\1}.\\rlap{\2}', '{:0.2%}'.format(x)).replace('%', ''))
    df.loc['$|\hat{Y}|$'] = df.loc['$|\hat{Y}|$'].apply(
        lambda x: re.sub(r'(\d+)\.(\d+)', r'\\llap{\1}.\\rlap{\2}', '{:0.2f}'.format(x)))
    df = df.stack(0)[['Known', 'Unknown', 'Total']]

    with open(outdir + savefile[:-3] + 'tex', 'w') as f:
        df.unstack(0).to_latex(buf=f, escape=False, multicolumn_format='c', multirow=True,
                               column_format='l@{\enspace}c@{\enspace}c@{\enspace}c@{\qquad}c@{\enspace}c@{\quad}c@{\qquad}c@{\enspace}c@{\enspace}c')
        # header=[r'\makebox[\widthof{Unknown}]{' + col + '}' for col in df.columns])
