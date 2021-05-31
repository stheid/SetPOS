import tempfile
from io import StringIO

import pandas as pd
from pie.scripts import train as pietrain
from pie.scripts import tag as pietag
from pie.settings import settings_from_file
from sklearn.model_selection import cross_val_score

from setpos.data.split import MCInDocSplitter, KFoldInDocSplitter, load, dump as split_dump
from setpos.tagger.base import StateFullTagger


class PieTagger(StateFullTagger):
    def __init__(self, augment_setvalued_targets=False, set_valued=True):
        super().__init__(set_valued)

        self.augment_setvalued_targets = augment_setvalued_targets

        self._tags.append('SENT')
        self.modelfile = None

    def fit(self, X, y):
        super().fit(X, y)

        with tempfile.NamedTemporaryFile('w', suffix='.tsv') as train:
            io = StringIO()
            split_dump([(X, y)], [io], as_dataset=True, augment=self.augment_setvalued_targets,
                       tagsed_sents_tostr_kws=dict(dlm=",", word_dlm='\n', sent_dlm='\n.\,SENT\n'))
            io.seek(0)

            df = (
                pd.read_csv(io, header=None, na_filter=False)
                    .rename(columns={0: 'token', 1: 'pos'})
                    .assign(lemma=lambda x: x.token)[['token', 'lemma', 'pos']]
            )
            df.to_csv(train, index=False)
            train.flush()

            settings = settings_from_file('settings.json')
            settings.input_path = train.name
            pietrain.run(settings)

    def predict_proba(self, X, y=None):
        with tempfile.NamedTemporaryFile('w', suffix='.tsv') as train:
            io = StringIO()
            split_dump([(X)], [io], as_dataset=True, augment=self.augment_setvalued_targets,
                       tagsed_sents_tostr_kws=dict(dlm=",", word_dlm='\n', sent_dlm='\n.\,SENT\n'))
            io.seek(0)

            df = (
                pd.read_csv(io, header=None, na_filter=False)
                    .rename(columns={0: 'token', 1: 'pos'})
                    .assign(lemma=lambda x: x.token)[['token', 'lemma', 'pos']]
            )
            df.to_csv(train, index=False)
            train.flush()

            settings = settings_from_file('settings.json')
            settings.input_path = train.name
            pietag.run(settings)


if __name__ == '__main__':
    toks, tags, groups = [l[:100] for l in load()]  # load()

    train, test = next(MCInDocSplitter(seed=1).split(toks, tags, groups))
    # train, test = next(LeaveOneGroupOut().split(toks, tags, groups))

    clf = PieTagger()
    clf.fit(toks[train], tags[train])
    print(f'meansetsize: {clf.meansetsize(toks[test]):.2f}')
    print(f'knownwords: {clf.knownwords(toks[test]):.2%}')
    print(
        f'accuracy: {cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(5, seed=1), n_jobs=1).mean():.2%}')
    clf.set_valued = True
    print(f'utility: {cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(5, seed=1), n_jobs=1).mean():.2%}')
