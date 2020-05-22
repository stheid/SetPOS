import json
import re
from collections import Counter
from itertools import chain

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from natsort import natsorted

from setpos.data.split import load

if __name__ == '__main__':
    _, tags, groups = load()
    docs = natsorted(set(groups), key=lambda x: str(int(not x.startswith('REN'))) + x)

    result = {}
    for doc in docs:
        t = list(chain.from_iterable([json.loads(tags_).keys() for tags_ in tags[groups == doc]]))
        counter = pd.Series(dict(Counter(t)))
        counter /= len(t)
        result[doc] = counter
    df = pd.DataFrame(result).sort_index().T

    t = list(chain.from_iterable([json.loads(tags_).keys() for tags_ in tags]))
    prior = pd.Series(dict(Counter(t))).sort_index()
    prior /= prior.sum()

    doc_sizes = pd.Series([len(tags[groups == doc]) for doc in docs], index=docs)
    prior_df = prior.sort_values(ascending=False).head(10)
    prior_df.name = 'Probability'
    prior_df = prior_df.to_frame().T
    percent_to_str = lambda x: f'{x:.2%}'.replace('%', '\%')
    convert_sc = lambda x: re.sub(r'([A-Z]+)', lambda x: f'\\textsc{{{x.group().lower()}}}', x)
    prior_df.columns = [convert_sc(col) for col in prior_df]
    print(prior_df.to_latex(float_format=percent_to_str, escape=False))

    with_prior = df
    df = df - prior

    fig, ((ax_prior, ax_space, ax_space2),
          (ax_main, ax_docsize, ax_cbar)) = plt.subplots(nrows=2, ncols=3,  # sharey='row',
                                                         gridspec_kw={"height_ratios": (1, 14), "hspace": .05,
                                                                      "width_ratios": (30, 1, 1), "wspace": .05})
    fig.set_size_inches((16, 9.6))

    # shows prior tag distribution
    prior.plot.bar(ax=ax_prior)
    ax_prior.get_xaxis().set_visible(False)

    # plot top down
    doc_sizes.iloc[::-1].plot.barh(ax=ax_docsize)
    ax_docsize.get_yaxis().set_visible(False)

    # each row shows difference in probablity mass for one document vs the global prior
    sns.heatmap(df, xticklabels=1, yticklabels=1, cmap='PRGn', edgecolors='white', linewidths=1, ax=ax_main,
                cbar_ax=ax_cbar, center=0, fmt='3.0%', annot=with_prior,
                annot_kws=dict(fontsize="xx-small", rotation='vertical'))
    for text in ax_main.texts:
        text.set_visible(float(text._text[:-1]) >= 1)

    # https://stackoverflow.com/a/56942725 fix for matplotlib 3.1.1
    ax_main.set_ylim(23, 0)

    # the NaN values are basically the points where the negative difference is maximal, because the frequency of the tag is zero
    ax_main.patch.set(hatch='xxx', ec='#40004b', alpha=.5)

    # remove padding fields in the top right corner
    ax_space.set_visible(False)
    ax_space2.set_visible(False)
    fig.savefig("dataset_tagging_differences.pdf", bbox_inches='tight')

    # df = pd.DataFrame(result).fillna(0)
    # fig = sns.clustermap(df.T, xticklabels=1, cmap='hot',col_cluster=False).savefig("out", bbox_inches='tight')
