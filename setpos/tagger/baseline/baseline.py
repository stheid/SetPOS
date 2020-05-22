import json

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score

from setpos.data.split import MCInDocSplitter, KFoldInDocSplitter
from setpos.data.split import load
from setpos.tagger.base import BaseTagger


class MostFrequentTag(BaseTagger):
    def __init__(self, set_valued=False):
        super().__init__()
        self.model = None
        self.set_valued = set_valued
        self._tags = None
        self._words = None

    def fit(self, X, y):
        orig = pd.DataFrame(np.vstack((X[:, 1], y)).T, columns=['x', 'y'])
        orig.y = orig.y.apply(json.loads)
        df = pd.DataFrame([pd.Series(entry) for entry in orig.y.values]).fillna(0)
        prior = df.sum()
        self._tags = list(prior.index)

        df = pd.concat([df, orig.x], axis=1).groupby('x').sum()
        df = pd.concat([df, pd.DataFrame(prior, columns=['<prior>']).T])  # type:pd.DataFrame
        self.model = df.div(df.sum(axis=1), axis=0)
        self._words = set(self.model.index)

    def predict_proba(self, X, **kwargs):
        preds = []
        for tok in X[:, 1]:
            if tok in self._words:
                # probabilities in model, are sorted by self._tags
                preds.append(self.model.loc[tok].to_numpy())
            else:
                preds.append(self.model.iloc[-1].to_numpy())
        return np.array(preds)

    def predict(self, X):
        if self.model is None:
            raise NotFittedError
        return super(MostFrequentTag, self).predict(X)


if __name__ == '__main__':
    toks, tags, groups = load()

    train, test = next(MCInDocSplitter(seed=1).split(toks, tags, groups))
    # train, test = next(LeaveOneGroupOut().split(toks, tags, groups))

    clf = MostFrequentTag()
    clf.fit(toks[train], tags[train])
    print(f'meansetsize: {clf.meansetsize(toks[test]):.2f}')
    print(f'knownwords: {clf.knownwords(toks[test]):.2%}')
    print(f'accuracy: {cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(5), n_jobs=3).mean():.2%}')
    clf.set_valued = True
    print(f'utility: {cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(5), n_jobs=4).mean():.2%}')

"""
> meansetsize: 4.09
> knownwords: 92.79%
> accuracy: 74.48%
> utility: 92.88%
"""
