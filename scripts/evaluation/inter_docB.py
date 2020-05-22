import re
from collections import Counter
from os.path import isfile

import numpy as np
import pandas as pd

from scripts.evaluation.general import SEED, clfs, evaluate, plot_dataframe
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
    for doc, _ in filter(lambda t: t[0] in ['DSR', 'REN4', 'REN14', 'Ko'], ctr.items()):
        mask = groups_t != doc
        mask_e = groups_e == doc
        Xtrain, ytrain = toks_t[mask], tags_t[mask]
        Xeval, yeval = toks_e[mask_e], tags_e[mask_e]
        print(doc, np.sum(mask), np.sum(mask_e))
        maskedtoks, maskedtags = np.vstack((Xtrain, Xeval)), np.append(ytrain, yeval)
        for clf in clfs:
            clfname = 'baseline' if clf.get('clf', None) == MostFrequentTag else (
                'TreeTagger' if clf.get('clf', None) == TreeTagger else r'\textsc{c}ore\textsc{nlp}')

            istrain = np.append(np.full(ytrain.shape, True), np.full(yeval.shape, False))

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
    savefile = 'inter-docB.pkl'
    if isfile(savefile):
        df = pd.read_pickle(savefile)
    else:
        d = calc_results()
        df = pd.DataFrame(d, index=('\mlacc', '\mlutil', '$|\hat{Y}|$'))
        df.to_pickle(savefile)

    plot_dataframe(df, savefile)
