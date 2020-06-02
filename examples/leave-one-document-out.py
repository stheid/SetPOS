import logging

from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

from setpos.tagger import CoreNLPTagger, TreeTagger
from setpos.data.split import load

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    toks, tags, groups = load(n=5000)

    clf = CoreNLPTagger()
    scores = cross_val_score(clf, toks, tags, groups, cv=LeaveOneGroupOut(), n_jobs=-1)

    print(f"total accuracy: {scores.mean():.2%} ± {scores.std():.2%}")
