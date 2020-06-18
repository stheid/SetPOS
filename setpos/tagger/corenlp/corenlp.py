import contextlib
import json
import logging
import os
import re
import tempfile
from subprocess import run

import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.model_selection import cross_val_score

from setpos.data.evaluate import print_classical_pred_stats, print_set_valued_pred_stats, load as eval_load
from setpos.data.split import MCInDocSplitter, KFoldInDocSplitter, dump as split_dump, is_masked
from setpos.tagger.base import StateFullTagger
from setpos.util import create_ubop_predictor

logger = logging.getLogger(__name__)


class CoreNLPTagger(StateFullTagger):
    jar_dir = os.path.dirname(os.path.abspath(__file__))
    classpath = ':'.join([os.path.join(jar_dir, 'stanford-corenlp.jar')])
    clsname = 'edu.stanford.nlp.tagger.maxent.MaxentTagger'

    def __init__(self, corenlp_train_params=None, corenlp_infer_params=None, memlimit=None, loglevel=None,
                 augment_setvalued_targets=False, set_valued=False, tag_expansion_rules=(), timeout=1200):
        super().__init__(set_valued)
        self.corenlp_train_params = corenlp_train_params
        self.corenlp_infer_params = corenlp_infer_params
        self.memlimit = memlimit
        assert isinstance(augment_setvalued_targets, bool)
        self.augment_setvalued_targets = augment_setvalued_targets
        self.loglevel = loglevel or logging.WARN
        self.tag_expansion_rules = tag_expansion_rules
        self.timeout = timeout
        self.set_valued = set_valued

        self.train_params = self.corenlp_train_params or []
        self.infer_params = self.corenlp_infer_params or []
        self.javaops = f'-Xmx{self.memlimit or "8g"}'
        self.error = ''
        logger.setLevel(self.loglevel)

    def fit(self, X, y, **kwargs):
        super().fit(X, y)

        with tempfile.NamedTemporaryFile("w") as f:
            split_dump([(X, y)], [f], as_dataset=True, augment=self.augment_setvalued_targets)
            f.flush()

            result = run(['java', self.javaops, '-cp', self.classpath, self.clsname,
                          '-props', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training.props'),
                          '-model', self.modelfile,
                          '-trainFile', f.name] + self.train_params, capture_output=True)
            self.error = result.stderr.decode('utf8')
            if result.returncode != 0:
                print(result.args)
                exception_text = re.search("Exception in [\s\S]*", self.error, re.MULTILINE)
                print(exception_text[0] if exception_text else self.error)
                raise RuntimeError()
        logger.info('fitted model')
        return self

    def transform(self, X):
        super().transform(X)

        with tempfile.NamedTemporaryFile("w") as f, tempfile.TemporaryDirectory() as dir:
            split_dump([(X,)], [f], as_dataset=True)
            f.flush()

            if self.tag_expansion_rules:
                with tempfile.NamedTemporaryFile("w", delete=False, dir=dir) as expansionRuleFile:
                    print('\n'.join([','.join(rule) for rule in self.tag_expansion_rules]), file=expansionRuleFile)
                    f.flush()
                    self.infer_params.extend(
                        ['-doDeterministicTagExpansion', '-tagExpansionRuleFile', expansionRuleFile.name])

            result = run(['java', self.javaops, '-cp', self.classpath, self.clsname,
                          '-model', self.modelfile,
                          '-testFile', f.name,
                          '-debugPrefix', dir,
                          '-debug'] + self.infer_params, capture_output=True, timeout=self.timeout)
            self.error = result.stderr.decode('utf8')
            if result.returncode != 0:
                print(result.args)
                exception_text = re.search("Exception in [\s\S]*", self.error, re.MULTILINE)
                print(exception_text[0] if exception_text else self.error)
                raise RuntimeError()
            try:
                df, tags = eval_load(dir)
                df.drop(['target'], axis=1)
                logger.info('transformed data')
                return df, tags
            except EmptyDataError:
                print(self.error)
                raise RuntimeError()

    def singlepredict(self, X):
        df, tags = self.transform(X)
        return df.pred.to_numpy()

    def predict_proba(self, X, y=None):
        df, tags = self.transform(X)
        return np.exp(np.array(df.posterior.tolist())), tags

    def setpredict(self, X):
        """
        :param X:
        :return: (possibly weighted) set as a json string of OrderedDict
        """
        df, tags = self.transform(X)
        set_predictor = create_ubop_predictor(self.g, is_weigthed=True)
        df = df.assign(constrained_posterior=df.apply(
            lambda x: x.posterior[[tags.inv[t] for t in x.constrainedtags]], axis=1))
        df = df.assign(set_pred=df.apply(
            lambda x: set_predictor(x['constrained_posterior'], x.constrainedtags, True), axis=1))
        # np.array of jsonified dicts
        return df.set_pred.apply(json.dumps).to_numpy()

    def score(self, X, y, raw=False, long_result=False, score=None, scope='total', eval_mask=('$')):
        df, tags = self.transform(X)
        df = df.assign(target=y)

        # filter evaluation to non-orthographic tags
        df = df[df.target.apply(lambda x: is_masked(x, eval_mask))]

        if raw:
            return df, tags

        with contextlib.redirect_stdout(None):
            df, results = print_classical_pred_stats(df, tags)
            df, results2 = print_set_valued_pred_stats(df, tags, g=self.g)
        results = pd.concat([results, results2])
        if long_result:
            return df, results
        if score is None:
            score = 'accuracy' if self.set_valued == False else 'avg util'
        return results[scope].loc[score]


if __name__ == '__main__':
    import numpy as np
    from setpos.data.split import load, sents_to_dataset
    import logging

    logging.basicConfig()

    toks, tags, groups = load()
    g = list(set(groups))
    clf = CoreNLPTagger(loglevel=logging.INFO)

    clf.fit(toks[np.isin(groups, g[1:2])], tags[np.isin(groups, g[1:2])])
    clf.fit(toks[np.isin(groups, g[0:1])], tags[np.isin(groups, g[0:1])])
    # clf.fit(toks, tags)

    # schap/NA vnde/KON dar/PAVD to/PAVAP hebbe/VAFIN ik/PPER ere/PPER gegeuen/VVPP
    print(clf.predict_proba(sents_to_dataset(['schap vnde dar to hebbe ik ere gegeuen'.split()])))
    print(clf.setpredict(sents_to_dataset(['schap vnde dar to hebbe ik ere gegeuen'.split()])))

    results = clf.score(toks[np.isin(groups, g[1:2])], tags[np.isin(groups, g[1:2])], raw=True)
    print_classical_pred_stats(*results)

    toks, tags, groups = load()

    train, test = next(MCInDocSplitter(seed=1).split(toks, tags, groups))

    clf = CoreNLPTagger(set_valued=False, corenlp_train_params=['--arch',
                                                                'left5words,suffix(1),prefix(1),suffix(2),prefix(2),suffix(3),prefix(3)'])
    clf.fit(toks[train], tags[train])

    print(f'meansetsize: {clf.meansetsize(toks[test]):.2f}')
    print(
        f'accuracy: {cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(5, seed=1), n_jobs=-1).mean():.2%}')
    clf.set_valued = True
    print(
        f'utility: {cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(5, seed=1), n_jobs=-1).mean():.2%}')

'''
> meansetsize: 2.71
> accuracy: 84.49%
> utility: 94.87%
'''
