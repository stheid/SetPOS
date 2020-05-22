import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from setpos.tagger.base import BaseTagger
from setpos.data.split import MCInDocSplitter, load, join_to_sents, KFoldInDocSplitter


class SimpleTagger(BaseTagger):
    def __init__(self, set_valued=False):
        super().__init__()
        self.clf = None

        self.set_valued = set_valued

    @staticmethod
    def _features(sentence, index):
        """ sentence: [w1, w2, ...], index: the index of the word """
        return {
            'word': sentence[index],
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            'is_capitalized': sentence[index][0].upper() == sentence[index][0],
            'is_all_caps': sentence[index].upper() == sentence[index],
            'is_all_lower': sentence[index].lower() == sentence[index],
            'prefix-1': sentence[index][0],
            'prefix-2': sentence[index][:2],
            'prefix-3': sentence[index][:3],
            'suffix-1': sentence[index][-1],
            'suffix-2': sentence[index][-2:],
            'suffix-3': sentence[index][-3:],
            'prev_word': '' if index == 0 else sentence[index - 1],
            'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
            'prevprev_word': '' if index <= 1 else sentence[index - 2],
            'nextnextnext_word': '' if index >= len(sentence) - 2 else sentence[index + 2],
            'has_hyphen': '-' in sentence[index],
            'is_numeric': sentence[index].isdigit(),
            'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
        }

    def fit(self, X, y):
        self._words = set(X[:, 1])
        sents = join_to_sents(X, y)
        newX, newy = [], []
        for sent in sents:
            for index in range(len(sent)):
                newX.append(self._features([w for w, t in sent], index))
                newy.append(list(sent[index][1].keys())[0])

        self.clf = Pipeline([
            ('vectorizer', DictVectorizer(sparse=True)),
            ('classifier', LogisticRegression(solver='lbfgs', multi_class='auto'))
        ])
        self.clf.fit(newX, newy)
        return self

    def predict_proba(self, X, y=None):
        sents = join_to_sents(X)
        newX = []
        for sent in sents:
            for index in range(len(sent)):
                newX.append(self._features(sent, index))
        probas = self.clf.predict_proba(newX, )
        # pad the probabilities with zero rows for every tag that has been unseen during training
        tag_idx = {t: i for i, t in enumerate(self.clf.classes_)}
        new_probas = np.zeros((probas.shape[0], len(self._tags)))
        for i, t in enumerate(self._tags):
            if t in tag_idx:
                new_probas[:, i] = probas[:, tag_idx[t]]
        return new_probas

    def predict(self, X):
        if self.clf is None:
            raise NotFittedError
        return super().predict(X)


if __name__ == '__main__':
    toks, tags, groups = [l[:] for l in load()]  # load()

    train, test = next(MCInDocSplitter(seed=1).split(toks, tags, groups))
    # train, test = next(LeaveOneGroupOut().split(toks, tags, groups))

    clf = SimpleTagger()
    clf.fit(toks[train], tags[train])
    print(f'meansetsize: {clf.meansetsize(toks[test]):.2f}')
    print(f'knownwords: {clf.knownwords(toks[test]):.2%}')
    print(
        f'accuracy: {cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(5, seed=1), n_jobs=3).mean():.2%}')
    clf.set_valued = True
    print(f'utility: {cross_val_score(clf, toks, tags, groups, cv=KFoldInDocSplitter(5, seed=1), n_jobs=4).mean():.2%}')

"""
> meansetsize: 3.50
> knownwords: 91.71%
> accuracy: 81.14%
> utility: 94.15%
"""
