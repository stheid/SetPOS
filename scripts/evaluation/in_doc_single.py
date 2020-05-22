import re
from collections import Counter
from os.path import isfile

import numpy as np
import pandas as pd

from scripts.evaluation.general import SEED, clfs, evaluate, plot_dataframe
from setpos.data.split import load, MCInDocSplitter
from setpos.tagger import MostFrequentTag, TreeTagger


def calc_results():
    toks, tags, groups = load(tag_prefix_masks=[])  # [l[:3000] for l in load(tag_prefix_masks=[])]  #
    ctr = Counter(groups)

    # train - test split
    train, test = next(MCInDocSplitter(seed=SEED).split(toks, tags, groups))

    # take the training data for train/eval cross-validation
    Xtrain, ytrain, _ = [l[train] for l in [toks, tags, groups]]

    # take the training data for train/eval cross-validation
    toks_e, tags_e, groups_e = [l[test] for l in [toks, tags, groups]]

    d = {}
    for clf in clfs:
        old_clf = None
        clfname = 'baseline' if clf.get('clf', None) == MostFrequentTag else (
            'TreeTagger' if clf.get('clf', None) == TreeTagger else r'\textsc{c}ore\textsc{nlp}')
        for doc, _ in filter(lambda t: t[1] > 7000, ctr.items()):
            mask_e = groups_e == doc
            print(doc, ytrain.shape[0], np.sum(mask_e))
            Xeval, yeval = toks_e[mask_e], tags_e[mask_e]
            maskedtoks, maskedtags = np.vstack((Xtrain, Xeval)), np.append(ytrain, yeval)
            istrain = np.append(np.full(ytrain.shape, True), np.full(yeval.shape, False))

            doc = ''.join(((r'\textsc{' + c.lower() + '}' if re.match('[A-Z]', c) != None else c) for c in doc))
            # two classifiers, k datasets, 3 metrics -> two grouped barplots (over datasets), one for acc;util
            #           table (clf row, dataset col, subtable in cell for each metric)
            old_clf, scores = evaluate(clf, maskedtoks, maskedtags, istrain, oldclf=old_clf)
            d[(doc, clfname)] = scores
    print()
    return d


if __name__ == '__main__':
    savefile = 'in-doc-single.pkl'
    if isfile(savefile):
        df = pd.read_pickle(savefile)
    else:
        d = calc_results()
        df = pd.DataFrame(d, index=('\mlacc', '\mlutil', '$|\hat{Y}|$'))
        df.to_pickle(savefile)

    plot_dataframe(df, savefile, setsize_ticks_sep=1, setsize_ylim=dict(bottom=None, top=5))

    args = dict(wspace=(.1, .15), legend_pos='upper right', setsize_ticks_sep=.5, acc_ticks_sep=.06,
                acc_ylim=dict(bottom=-.06, top=.37), setsize_ylim=dict(bottom=-2, top=.5))

    df_diff = df - pd.read_pickle('in-domain.pkl')
    plot_dataframe(df_diff, savefile[:-4] + '—' + 'in-domain.pkl', **args)

    df_diff = df - pd.read_pickle('inter-docB.pkl')
    plot_dataframe(df_diff, savefile[:-4] + '—' + 'inter-docB.pkl', **args)
