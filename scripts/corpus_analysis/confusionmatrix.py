import json

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from setpos.tagger import CoreNLPTagger
from setpos.data.split import MCInDocSplitter, load
from setpos.util import stopwatch

if __name__ == '__main__':
    SEED, n = 7, 6
    toks, tags, groups = load()  # [l[:3000] for l in load()]  #

    with stopwatch():
        targets = []
        preds = []
        for train, test in MCInDocSplitter(seed=SEED).split(toks, tags, groups):
            clf = CoreNLPTagger()
            clf.fit(toks[train], tags[train])
            targets.extend([list(json.loads(tags_).keys())[0] for tags_ in tags[test]])
            preds.extend(clf.predict(toks[test]))

    tags = set(targets) | set(preds)
    print(len(tags))
    conf_mat = confusion_matrix(targets, preds)
    with np.errstate(divide='ignore'):
        conf_mat = conf_mat / conf_mat.sum(axis=1)

    df = pd.DataFrame(conf_mat.T, columns=sorted(tags), index=sorted(tags))

    # y axis is true, x axis is pred
    ax = sns.heatmap(df, cmap='Greens', vmin=0, vmax=1, xticklabels=1, yticklabels=1, edgecolors='white', linewidths=1)
    ax.patch.set(hatch='xxx', ec='.8')
    ax.set_ylabel('true')
    ax.set_xlabel('pred')

    fig = ax.get_figure()
    fig.set_size_inches((16, 14))
    fig.savefig("confusion.pdf", bbox_inches='tight')
