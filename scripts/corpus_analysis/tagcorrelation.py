"""
Calculate the similarity of tags by calculation of the jaccard similarity of the words it appears with
"""

import json
from collections import defaultdict, Counter
from operator import itemgetter

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from setpos.data.split import load


def weighted_jaccard(x, y):
    # Ruzicka similarity
    sim = sum((min(*pair) for pair in zip(x, y))) / sum((max(*pair) for pair in zip(x, y)))
    return 0 if sim == np.NaN else sim


if __name__ == '__main__':
    X, y, g = load()

    # dict of tag -> set of words
    tag_words_map = defaultdict(set)
    tagcount = Counter()
    for tags, (_, word) in zip(y, X):
        tags = json.loads(tags).keys()
        tagcount.update(tags)
        for tag in tags:
            tag_words_map[tag].add(word)
    tagcount = dict(tagcount)

    # distance matrix
    # dists = np.empty((len(tag_words_map), len(tag_words_map)))
    #
    # pipeline = Pipeline([('vectorizer', CountVectorizer(binary=True)),
    #                      ('todense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
    #                      ('tSNE', TSNE(metric=lambda x, y: 1 - jaccard_score(x, y), random_state=0, perplexity=20))])
    #
    # positions = pipeline.fit_transform([' '.join(list(s)) for s in tag_words_map.values()])
    # df = pd.DataFrame(positions, columns=['x', 'y'])
    # df = df.assign(tag=list(tag_words_map.keys()))
    # df = df.assign(count=df.tag.apply(lambda x: tagcount[x]))
    #
    # ax = sns.scatterplot(x='x', y='y', size='count', data=df)
    #
    # for _, r in df.iterrows():
    #     ax.text(r.x, r.y, r.tag, fontsize=8, horizontalalignment='center', verticalalignment='center')
    #
    # fig = ax.get_figure()
    # fig.set_size_inches((12, 8))
    # fig.savefig("tag_correlation.pdf")
    # plt.clf()

    pipeline = Pipeline([('vectorizer', CountVectorizer(binary=False)),
                         ('todense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
                         ('dists', FunctionTransformer(
                             lambda e: pairwise_distances(e, metric=lambda x, y: weighted_jaccard(x, y))))])

    dists = pipeline.fit_transform([' '.join(list(s)) for s in tag_words_map.values()])
    df = pd.DataFrame(dists, columns=tag_words_map.keys(), index=tag_words_map.keys())
    print(sorted(df.NA.to_dict().items(), key=itemgetter(1), reverse=True))
    print('APPR', (df - np.diag(np.diag(df.to_numpy()))).APPR.idxmax(),
          (df - np.diag(np.diag(df.to_numpy()))).APPR.max())
    sns.clustermap(df, xticklabels=1, yticklabels=1, figsize=(14, 14)).savefig('distances.pdf')

    # remove identity distance pairs, remove duplicates from symmetric relation by sorting keys and putting them into a dict
    tag_pairs_by_dist = pd.Series(
        {tuple(sorted(k)): v for k, v in df.unstack().to_dict().items() if k[0] != k[1]}).sort_values(
        ascending=False).to_frame()
    tag_pairs_by_dist.columns = ['Ruzicka']
    tag_pairs_by_dist.to_csv('tag_distance.csv')
    with open('tag_distance.tex', 'w') as f:
        tag_pairs_by_dist.head(10).T.to_latex(f)
