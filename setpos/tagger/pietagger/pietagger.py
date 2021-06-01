import tempfile
from contextlib import redirect_stdout, redirect_stderr
from glob import glob
from io import StringIO
from itertools import repeat
from os import remove, devnull
from os.path import basename, dirname

import pandas as pd
import torch.nn.functional as F
from pie import utils, pack_batch, BaseModel
from pie.scripts import train as pietrain
from pie.settings import settings_from_file
from sklearn.model_selection import cross_val_score

from setpos.data.split import MCInDocSplitter, KFoldInDocSplitter, load, dump as split_dump
from setpos.tagger.base import StateFullTagger


class PieTagger(StateFullTagger):
    def __init__(self, augment_setvalued_targets=False, set_valued=False):
        super().__init__(set_valued)

        self.augment_setvalued_targets = augment_setvalued_targets

        self._tags.append('SENT')
        self.modelfile = None

    def fit(self, X, y):
        super().fit(X, y)

        with tempfile.NamedTemporaryFile('w', suffix='.csv') as train:
            io = StringIO()
            split_dump([(X, y)], [io], as_dataset=True, augment=self.augment_setvalued_targets,
                       tagsed_sents_tostr_kws=dict(dlm='\t', word_dlm='\n', sent_dlm='\n.\,SENT\n'))
            io.seek(0)

            df = (
                pd.read_csv(io, header=None, na_filter=False, sep='\t')
                    .rename(columns={0: 'token', 1: 'pos'})
                #                .assign(lemma=lambda x: x.token)[['token', 'lemma', 'pos']]
            )
            df.to_csv(train, index=False)
            train.flush()

            settings = settings_from_file('settings.json')
            settings.input_path = train.name
            settings.modelpath = dirname(self.modelfile)
            settings.modelname = basename(self.modelfile)

            pietrain.run(settings)

            # pie appends a suffix to the modelfile. we will therefore delete the empty modelfile
            # and update the reference to the suffixed file
            remove(self.modelfile)
            self.modelfile = glob(self.modelfile + '*')[-1]

    def predict_proba(self, X, y=None):
        io = StringIO()
        split_dump([(X,)], [io], as_dataset=True, augment=self.augment_setvalued_targets,
                   tagsed_sents_tostr_kws=dict(tags_to_string=None, word_dlm=' ', sent_dlm=' '))
        io.seek(0)
        settings = settings_from_file('settings.json')

        lines = io.readlines()
        with redirect_stdout(open(devnull, 'w')), redirect_stderr(open(devnull, 'w')):
            model = BaseModel.load(self.modelfile)
            tags = model.label_encoder.tasks['pos'].inverse_table

            probas = []
            for chunk in utils.chunks(lines, settings.batch):
                batch = list(zip(chunk, repeat(None)))

                inp, _ = pack_batch(model.label_encoder, batch, settings.cpu)

                (word, wlen), (char, clen) = inp
                emb, *_ = model.embedding(word, wlen, char, clen)

                enc_outs = model.encoder(emb, wlen)
                outs = enc_outs[-1]

                decoder = model.decoders['pos'](outs)
                proba = F.softmax(decoder, dim=-1)
                probas.append(proba)

            # return dense matrix of probabilities, each row expresses probabilities of the tags in self._tags sorted
            # one row per X

            # https://github.com/emanjavacas/pie/blob/0ca4311a57b2439994e5fcdc02f4d008ee268a9c/pie/models/decoder.py#L75
            df = pd.DataFrame(probas[0].detach().numpy().squeeze()[:len(X), :], columns=tags)
        return pd.DataFrame(df, columns=sorted(self._tags)).fillna(0).to_numpy(dtype=float)


if __name__ == '__main__':
    toks, tags, groups = [l[:] for l in load()]  # load()

    train, test = next(MCInDocSplitter(seed=1).split(toks, tags, groups))
    # train, test = next(LeaveOneGroupOut().split(toks, tags, groups))

    clf = PieTagger()
    clf.fit(toks[train], tags[train])

    mean_setsize = clf.meansetsize(toks[test])
    print(f'meansetsize: {mean_setsize:.2f}')
    known_words = clf.knownwords(toks[test])
    print(f'knownwords: {known_words:.2%}')
    acc = clf.singlescore(toks[test], tags[test])
    print(f'accuracy: {acc:.2%}')
    util = clf.setscore(toks[test], tags[test])
    print(f'util: {util:.2%}')

    exit(0)
    accuracy = cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(5, seed=1), n_jobs=1).mean()
    print(f'accuracy: {accuracy:.2%}')
    utility = cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(5, seed=1), n_jobs=1).mean()

    print(f'meansetsize: {mean_setsize:.2f}')
    print(f'knownwords: {known_words:.2%}')
    clf.set_valued = False
    print(f'accuracy: {accuracy:.2%}')
    clf.set_valued = True
    print(f'utility: {utility:.2%}')
