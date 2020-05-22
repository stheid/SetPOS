from itertools import chain
from random import Random
from typing import Sequence, Tuple

import numpy as np

from setpos.data.split.io_ import load, join_to_sents, dump


class InDocSplitter:
    def __init__(self, randomized, seed=None):
        self.randomized = randomized
        self.rnd = Random(seed)

    def _get_docs(self, X, groups):
        X, groups = map(np.array, [X, groups])
        idxs = np.array([(sentid, idx) for idx, (sentid, word) in enumerate(X)], dtype=int)

        docs = {}
        for doc_key in sorted(set(groups)):
            sents = join_to_sents(idxs[groups == doc_key])
            if self.randomized:
                self.rnd.shuffle(sents)
            docs[doc_key] = sents
        return docs

    @staticmethod
    def _split_doc(cut1, cut2, tagged_sents=Sequence[Sequence[int]]) -> Tuple[
        Sequence[Sequence[int]], Sequence[Sequence[int]]]:
        """
        It will take a continuous part of tokens specified by the two cuts.
        The sentence boundaries are purposefully ignored.
        the sentences are provided as integers
        :param cut1:
        :param cut2:
        :param tagged_sents: sequence of token ids
        :return:
        """
        # this means we will flip cutting positions and also flip the returned arrays
        invert = cut1 > cut2
        if invert:
            cut1, cut2 = cut2, cut1
        outer, inner = [], []
        i = -1  # preincrement
        for sent in tagged_sents:
            new_sent = []
            for tok in sent:
                i += 1
                if i == cut1:
                    # first element in inner
                    if new_sent:
                        outer.append(new_sent)
                    new_sent = []
                if i == cut2:
                    # first element in outer
                    if new_sent:
                        inner.append(new_sent)
                    new_sent = []
                new_sent.append(tok)
            if new_sent:
                if cut1 <= i < cut2:
                    inner.append(new_sent)
                else:
                    outer.append(new_sent)
        return (outer, inner) if invert else (inner, outer)


class MCInDocSplitter(InDocSplitter):
    """
    Monte Carlo crossvalidation splitter
    """

    def __init__(self, train_frac=.8, splits=1, randomized=True, seed=5, skip_splits=0, **kwargs):
        super().__init__(randomized, seed, **kwargs)
        self.train_frac = train_frac
        self.splits = splits
        self.skip_splits = skip_splits

    def get_n_splits(self, *args, **kwargs):
        return max(0, self.splits - self.skip_splits)

    def split(self, X, y, groups):
        docs = self._get_docs(X, groups)

        for i in range(self.splits):
            train, eval = [], []
            for key, sents in docs.items():
                n_toks = sum(map(len, sents))

                cut1 = self.rnd.randrange(0, n_toks)
                cut2 = (cut1 + int(self.train_frac * n_toks)) % n_toks

                t, e = self._split_doc(cut1, cut2, sents)
                train.extend(chain(*t))
                eval.extend(chain(*e))

            if i >= self.skip_splits:
                # it is important that we still generate the skipped splits, otherwise the random generator will be screwed up
                # return train_set, test_set
                yield train, eval


class KFoldInDocSplitter(InDocSplitter):
    def __init__(self, k=5, randomized=True, **kwargs):
        super().__init__(randomized, **kwargs)
        self.k = k

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.k

    def split(self, X, y, groups):
        docs = self._get_docs(X, groups)

        cuts = {k: self.rnd.randrange(0, sum(map(len, sents))) for k, sents in docs.items()}

        for i in range(self.k):
            train, eval = [], []
            for k, sents in docs.items():
                n_toks = sum(map(len, sents))
                cut1 = cuts[k] + i * n_toks // self.k
                cut2 = (cut1 + n_toks - n_toks // self.k) % n_toks

                # split up train and test sets
                # join documents, such that we only have a list of sentences
                # convert list of (tok,tags) into tuple of lists
                t, e = self._split_doc(cut1, cut2, sents)
                train.extend(chain(*t))
                eval.extend(chain(*e))

            # return train_set, test_set
            yield train, eval


if __name__ == '__main__':
    X, y, groups = load()

    split = MCInDocSplitter()

    train, eval = next(split.split(X, y, groups))

    dump(((X, y), (X[train], y[train]), (X[eval], y[eval])), as_dataset=True)
