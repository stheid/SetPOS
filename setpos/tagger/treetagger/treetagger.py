import json
import tempfile
from collections import defaultdict
from os import path
from subprocess import run

import pandas as pd
from sklearn.model_selection import cross_val_score

from setpos.data.split import MCInDocSplitter, KFoldInDocSplitter, dump as split_dump, load
from setpos.tagger.base import StateFullTagger


class TreeTagger(StateFullTagger):
    def __init__(self, augment_setvalued_targets=False, set_valued=False, train_params=None, eval_params=None,
                 fitbinary: str = None, predictbinary: str = None):
        super().__init__(set_valued)
        self.train_params = train_params
        self.eval_params = eval_params
        self.augment_setvalued_targets = augment_setvalued_targets
        self.set_valued = set_valued

        self._train_params = self.train_params or []
        self._eval_params = self.eval_params or []

        # if no binaries are provided we assume they are laying same folder as this file
        self.fitbinary = fitbinary or path.join(path.dirname(path.abspath(__file__)), 'train-tree-tagger')
        self.predictbinary = predictbinary or path.join(path.dirname(path.abspath(__file__)), 'tree-tagger')

        self._tags.append('SENT')
        self.modelfile = None

    def _calculate_lexicon(self, X, y):
        # list of words with all its tags it appears with
        text = zip(X[:, 1], y)
        lexicon = defaultdict(set)
        for tok, tag in text:
            lexicon[tok] |= set(json.loads(tag).keys())
        # treetagger expects each tag in the lexicon
        lexicon['.'].add('SENT')
        for tag in self._tags:
            lexicon[tag].add(tag)

        return sorted(lexicon.items())

    def fit(self, X, y):
        super().fit(X, y)

        with tempfile.NamedTemporaryFile("w") as train, tempfile.NamedTemporaryFile(
                "w") as lex, tempfile.NamedTemporaryFile("w") as opentags:
            # lexicon
            for tok, tag in self._calculate_lexicon(X, y):
                print(tok, '\t'.join((f'{tag} -' for tag in tag)), sep='\t', file=lex)

            split_dump([(X, y)], [train], as_dataset=True, augment=self.augment_setvalued_targets,
                       tagsed_sents_tostr_kws=dict(dlm="\t", word_dlm='\n', sent_dlm='\n.\tSENT\n'))

            print(*[k for k in self._tags], sep=' ', file=opentags)
            lex.flush()
            train.flush()
            opentags.flush()

            result = run(
                [f'{self.fitbinary}', lex.name, opentags.name, train.name, self.modelfile] + self._train_params,
                capture_output=True)
            self.error = result.stderr.decode('utf8')

            if 'ERROR' in self.error:
                print(self.error)

    def predict_proba(self, X, y=None):
        with tempfile.NamedTemporaryFile("w") as eval, tempfile.NamedTemporaryFile("r") as out:
            split_dump([(X,)], [eval], as_dataset=True, augment=self.augment_setvalued_targets,
                       tagsed_sents_tostr_kws=dict(tags_to_string=None, word_dlm='\n', sent_dlm='.\n'))
            eval.flush()
            result = run(
                [f'{self.predictbinary}', '-prob', '-threshold', '0.0000000000000000000001', self.modelfile,
                 eval.name, out.name] + self._eval_params,
                capture_output=True)
            self.error = result.stderr.decode('utf8')
            if 'ERROR' in self.error:
                print(self.error)

            # return dense matrix of probabilities, each row expresses probabilities of the tags in self._tags sorted
            # one row per X
            entries = [dict([kv.split() for kv in l.strip().split('\t')]) for l in out.readlines()]
        return pd.DataFrame(entries, columns=sorted(self._tags)).fillna(0).to_numpy(dtype=float)


if __name__ == '__main__':
    toks, tags, groups = [l[:] for l in load()]  # load()

    train, test = next(MCInDocSplitter(seed=1).split(toks, tags, groups))
    # train, test = next(LeaveOneGroupOut().split(toks, tags, groups))

    clf = TreeTagger()
    clf.fit(toks[train], tags[train])
    print(f'meansetsize: {clf.meansetsize(toks[test]):.2f}')
    print(f'knownwords: {clf.knownwords(toks[test]):.2%}')
    print(
        f'accuracy: {cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(5, seed=1), n_jobs=1).mean():.2%}')
    clf.set_valued = True
    print(f'utility: {cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(5, seed=1), n_jobs=1).mean():.2%}')

'''
meansetsize: 1.96
knownwords: 91.71%
accuracy: 82.24%
utility: 93.26%
'''
