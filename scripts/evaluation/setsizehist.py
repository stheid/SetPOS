from os.path import isfile

import matplotlib.pyplot  as plt
import pandas as pd
import seaborn as sns

from scripts.evaluation.general import SEED, latexify
from setpos.data.split import MCInDocSplitter, load
from setpos.tagger import CoreNLPTagger


def calculate_setsizes():
    toks, tags, groups = load(tag_prefix_masks=[])

    # train - test split
    train, test = next(MCInDocSplitter(seed=SEED).split(toks, tags, groups))

    # take the training data for train/eval cross-validation
    Xtrain, ytrain, _ = [l[train] for l in [toks, tags, groups]]
    Xeval, yeval, _ = [l[test] for l in [toks, tags, groups]]

    clf = CoreNLPTagger()
    clf.fit(Xtrain, ytrain)

    return clf.setsizes(Xeval)


if __name__ == '__main__':
    savefile = __file__[:-3] + '.pkl'

    if isfile(savefile):
        df = pd.read_pickle(savefile)
    else:
        df = calculate_setsizes()
        df.to_pickle(savefile)

    sns.set(style="whitegrid")
    sns.violinplot(x=df.sizes, bw=.4, cut=0).get_figure().savefig('all.pdf')
    plt.clf()
    df.isknown = df.isknown.apply(lambda x: "known" if x else "unknown")
    df = df.rename(columns=dict(isknown='word known', sizes='set size'))
    sns.violinplot(data=df.assign(x=1), x='x', split=True, cut=0, scale="count", y='set size', hue='word known',
                   bw=.4).get_figure().savefig('knownword.pdf')

    latexify(columns=2)
    max_ = df['set size'].max()

    f, (ax, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[19000 - 18000, 7500], hspace=.1))
    ax.set_ylim(18000, 19000)  # outliers only
    ax2.set_ylim(0, 7500)  # most of the data

    # plot the same data on both axes
    df.pivot(columns='word known', values='set size')[['unknown', 'known']].plot.hist(ax=ax, stacked=True,
                                                                                      bins=max_ - 1, legend=False)
    df.pivot(columns='word known', values='set size')[['unknown', 'known']].plot.hist(ax=ax2, stacked=True,
                                                                                      bins=max_ - 1, legend=False)

    ax.set_xlim(1, 11)
    ax2.legend(loc=(.63, .55))

    ax.set_ylabel('')
    ax2.set_ylabel('')
    ax2.set_xlabel('Set size')
    f.suptitle('Histogram of set size')
    f.text(0.01, 0.5, 'Frequency', va='center', rotation='vertical')

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    r = 7500 / (19000 - 18000)
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (r * -d, r * +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (r * -d, r * +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    plt.subplots_adjust(left=.18)
    plt.show()
    f.savefig(__file__[:-3] + '.pdf')
