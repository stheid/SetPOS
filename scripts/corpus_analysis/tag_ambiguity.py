import json
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

from setpos.data.split import load

if __name__ == '__main__':
    X, y, g = load()

    # dict of tag -> set of words
    word_tags_map = defaultdict(set)
    for tags, (_, word) in zip(y, X):
        tags = json.loads(tags).keys()
        for tag in tags:
            word_tags_map[word].add(tag)

    series = pd.Series(word_tags_map).apply(len)
    print(series.describe())
    series.hist()
    plt.show()
