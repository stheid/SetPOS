import json
from collections import Counter

from setpos.data.split import load

if __name__ == '__main__':
    X, y, g = load()

    # dict of tag -> set of words
    tagweights = []
    for tags, (_, word), doc in zip(y, X, g):
        weights = tuple(sorted(json.loads(tags).values()))
        if weights not in {(1.0,), (1.0, 1.0), (0.5, 0.5), (0.33, 0.66)}:
            print(doc)
        tagweights.append(weights)
    ctr = Counter(tagweights)
    total = sum(ctr.values())
    for key in ctr.keys():
        ctr[key] = ctr[key] / total
    for k, v in ctr.items():
        print(f'{k}:{v:.2%}')
