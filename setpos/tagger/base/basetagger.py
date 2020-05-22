import contextlib
import json
import os
import tempfile
from functools import reduce
from operator import itemgetter, or_

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from setpos.tagger.base.functions import create_g_alpha_beta, util, create_ubop_predictor, score


class BaseTagger(BaseEstimator):
    N_TAGS = 92  # 93

    def __init__(self, set_valued=True):
        self.set_valued = set_valued
        self._tags = ['ADJA', 'ADJA<VVPP', 'ADJA<VVPS', 'ADJD', 'ADJN', 'ADJN<VVPP', 'ADJS', 'ADJV', 'APPO', 'APPR',
                      'AVD', 'AVG',
                      'AVNEG', 'AVREL', 'AVW', 'CARDA', 'CARDN', 'CARDS', 'DDA', 'DDART', 'DDD', 'DDN', 'DDS', 'DDSA',
                      'DGA', 'DGN',
                      'DGS', 'DIA', 'DIART', 'DID', 'DIN', 'DIS', 'DNEGA', 'DNEGS', 'DPDS', 'DPOSA', 'DPOSD', 'DPOSGEN',
                      'DPOSN',
                      'DPOSS', 'DRELA', 'DRELS', 'DWS', 'FM', 'KO*', 'KOKOM', 'KON', 'KOUS', 'NA', 'NE', 'OA', 'PAVAP',
                      'PAVD',
                      'PAVG', 'PAVREL', 'PG', 'PI', 'PKOR', 'PNEG', 'PPER', 'PRF', 'PTKA', 'PTKANT', 'PTKG', 'PTKN',
                      'PTKNEG',
                      'PTKREL', 'PTKVZ', 'PTKZU', 'VAFIN', 'VAFIN.*', 'VAFIN.ind', 'VAFIN.konj', 'VAINF', 'VAPP',
                      'VKFIN.*',
                      'VKFIN.ind', 'VKFIN.konj', 'VKINF', 'VKPP', 'VKPS', 'VMFIN.*', 'VMFIN.ind', 'VMFIN.konj', 'VMINF',
                      'VVFIN.*',
                      'VVFIN.ind', 'VVFIN.konj', 'VVIMP', 'VVINF', 'VVPP', 'VVPS', 'XY']
        self._words = None
        self.set_g()
        self.scope = 'total'

    def set_g(self, alpha=1, beta=1):
        self.g = create_g_alpha_beta(alpha, beta, self.N_TAGS)

    def fit(self, X, y):
        self._words = set(X[:, 1])
        self._tags = sorted(reduce(or_, [set(self._tags)] + [{k for k, v in json.loads(tags).items()} for tags in y]))

    def singlepredict(self, X):
        # get argmax prediction from each set for each token
        preds = pd.Series(self.setpredict(X))
        return preds.apply(lambda s: max(json.loads(s).items(), key=itemgetter(1))[0]).to_numpy(dtype=str)

    def predict_proba(self, X, y=None):
        pass

    def setpredict(self, X):
        """
        :param X:
        :return: (possibly weighted) set as a json string of OrderedDict
        """
        if self._tags is None:
            raise NotFittedError
        ubop = create_ubop_predictor(self.g, is_weigthed=True)
        probas = pd.DataFrame(self.predict_proba(X), columns=self._tags)
        return probas.apply(lambda row: json.dumps(ubop(row.to_numpy(), self._tags)), axis=1).to_numpy(dtype=str)

    def predict(self, X):
        return self.setpredict(X) if self.set_valued else self.singlepredict(X)

    def score(self, X, y):
        return self.setscore(X, y) if self.set_valued else self.singlescore(X, y)

    def meansetsize(self, X, y=None):
        df = pd.Series([set(json.loads(tags).keys()) for tags in self.setpredict(X)], name='pred').apply(len)
        df = self.filter_scope(X, df)
        return df.mean()

    def setsizes(self, X, y=None):
        df = pd.Series([set(json.loads(tags).keys()) for tags in self.setpredict(X)], name='pred').apply(len).to_frame(
            'sizes')
        isknown = pd.Series(X[:, 1]).apply(lambda tok: bool(tok in self._words))
        df = df.assign(isknown=isknown)
        return df

    def knownwords(self, X):
        return pd.Series(X[:, 1]).apply(lambda tok: int(tok in self._words)).mean()

    def filter_scope(self, X, df):
        if self.scope == 'total':
            return df
        mask = pd.Series(X[:, 1]).apply(lambda tok: bool(tok in self._words) == (self.scope == 'known'))
        return df.T[mask].T

    def setscore(self, X, y):
        preds = pd.Series([set(json.loads(tags).keys()) for tags in self.setpredict(X)], name='pred')
        targets = pd.Series([json.loads(tags) for tags in y], name='target')

        df = pd.DataFrame((preds, targets))
        df = self.filter_scope(X, df)
        return df.apply(lambda x: util(self.g, x.target, x.pred), axis=0).mean()

    def singlescore(self, X, y):
        preds = pd.Series(self.singlepredict(X), name='pred')
        targets = pd.Series([json.loads(tags) for tags in y], name='target')

        df = pd.DataFrame((preds, targets))
        df = self.filter_scope(X, df)
        return df.apply(lambda x: score(x.target, x.pred), axis=0).mean()


class StateFullTagger(BaseTagger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modelfile = None

    def fit(self, X, y):
        super().fit(X, y)
        self.cleanup(True)

    def transform(self, X):

        if self.modelfile is None:
            raise NotFittedError

    def cleanup(self, recreate=False):
        with contextlib.suppress(FileNotFoundError, AttributeError):
            if self.modelfile is not None:
                os.remove(self.modelfile)
                os.remove(self.modelfile + '.probs')

        self.modelfile = tempfile.mkstemp()[1] if recreate else None

    def __del__(self):
        self.cleanup()
