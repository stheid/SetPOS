import re
from functools import partial

import matplotlib
import numpy as np
from matplotlib import gridspec, pyplot as plt

from setpos.data.split import is_masked
from setpos.tagger import MostFrequentTag, TreeTagger, CoreNLPTagger

SEED = 1
clf = dict(augment_setvalued_targets=False, filter_tags=[], memlimit="32g",
           corenlp_train_params=["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1",
                                 "-rareWordMinFeatureThresh", "1", "-rareWordThresh", "6", "-sigmaSquared",
                                 "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch",
                                 "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wordTag(0,-1),words(-3,2)",
                                 '-lang', 'english'])
clfs = [clf, dict(clf=MostFrequentTag),
        dict(clf=TreeTagger,
             train_params=['-atg', '3.00532105520549', '-cl', '1', '-dtg', '0.07667067848725928',
                           '-ecw', '0.6844143695818022', '-lt', '0.018083566511875225', '-sw', '4.872697144433427'],
             eval_params=['-beam', '9.17490365933556e-05', '-eps', '0.9975176265361098'])]

outdir = '../../Papier/'

to_sc = lambda s: re.sub('[A-Z]+', lambda s_: f'\\textsc{{{s_[0].lower()}}}', s.title() if s == 'baseline' else s)


def latexify(fig_width=None, fig_height=None, height_mul: float = 1, width_mul: float = 1, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert (columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 2 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean * height_mul  # height in inches

    fig_width *= width_mul

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': [r'\usepackage{gensymb}'
                                      r'\usepackage{amsmath,amssymb,mathtools}'
                                      r'\newcommand{\mlutil}{\ensuremath{\operatorname{ml-util}}}'
                                      r'\newcommand{\mlacc}{\ensuremath{\operatorname{ml-acc}}}'],
              'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8,  # was 10
              'legend.fontsize': 8,  # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)


def evaluate(params, toks, tags, istrain, oldclf=None, betas=None, raw=False):
    if 'filter_tags' in params:
        mask = np.vectorize(partial(is_masked, prefixes=params['filter_tags']))(tags)
        toks, tags, istrain = [a[mask] for a in [toks, tags, istrain]]
    clf_class = params.get('clf', CoreNLPTagger)

    # remove higher level parameters
    params = {k: v for k, v in params.items() if k not in {'filter_tags', 'clf'}}
    if oldclf is not None:
        clf = oldclf
    else:
        clf = clf_class(**params)
        clf.fit(toks[istrain, :], tags[istrain])
        print(f'fitted {clf}')
    if not betas:
        if isinstance(clf, CoreNLPTagger):
            df, score = clf.score(toks[~istrain, :], tags[~istrain], long_result=True)
            if raw:
                return df, score
            scores = [score.loc[measure, 'total'] for measure in ['accuracy', 'avg util', 'avg setsize']]
        else:
            scores = [getattr(clf, func)(toks[~istrain, :], tags[~istrain]) for func in
                      ['singlescore', 'setscore', 'meansetsize']]
    else:
        if isinstance(clf, CoreNLPTagger):
            scores = []
            for i, beta in enumerate(betas):
                print(f'evaluating beta={beta} on {clf}')
                clf.set_g(beta=beta)
                df, score = clf.score(toks[~istrain, :], tags[~istrain], long_result=True)
                if raw:
                    return df, score
                measures = (['accuracy'] if i == 0 else []) + ['avg util', 'avg setsize']
                scores.extend([score.loc[measure, 'total'] for measure in measures])
        else:
            scores = [clf.singlescore(toks[~istrain, :], tags[~istrain])]
            for beta in betas:
                clf.set_g(beta=beta)
                scores.extend(
                    [getattr(clf, func)(toks[~istrain, :], tags[~istrain]) for func in ['setscore', 'meansetsize']])
    return clf, scores


def wavg(group, avg_name, weight_name):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean. Customize this if your business case
    should return otherwise.
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return d.multiply(w, axis=0).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


def plot_dataframe(df, savefile, wspace=(.1, .1), legend_pos='lower right', acc_ticks_sep=.1, setsize_ticks_sep=2,
                   acc_ylim=None, setsize_ylim=None):
    acc_ylim = acc_ylim or dict()
    setsize_ylim = setsize_ylim or dict()
    df = df.stack(1)
    df = df[[r'\textsc{d}\textsc{s}\textsc{r}', r'\textsc{r}\textsc{e}\textsc{n}4', r'\textsc{r}\textsc{e}\textsc{n}14',
             r'\textsc{k}o']]
    df.columns = ['Duisburg', 'Bremen', 'Bamberg', 'Kołobrzeg']
    df = df.unstack(-1)

    latexify(columns=1, width_mul=1.08, height_mul=.40)
    f = plt.figure()
    gs0 = gridspec.GridSpec(1, 2, figure=f, width_ratios=[2, 1], wspace=wspace[1])
    gs00 = gs0[0].subgridspec(1, 2, wspace=wspace[0])
    w = .66

    ax1 = f.add_subplot(gs00[0])
    ax2 = f.add_subplot(gs00[1])
    ax3 = f.add_subplot(gs0[1])

    ax = ax1
    df.loc['\mlacc'].unstack(-1)[['baseline', 'TreeTagger', r'\textsc{c}ore\textsc{nlp}']] \
        .rename(to_sc, axis='columns').plot.bar(ax=ax, legend=False, width=w)
    ax.tick_params(axis='x', rotation=20)
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y')
    ax.set_ylim(**(lambda d: d.update(acc_ylim) or d)(dict(bottom=.4, top=1)))
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end + acc_ticks_sep, acc_ticks_sep))
    ax.set_title('Accuracy')

    ax = ax2
    ax = df.loc['\mlutil'].unstack(-1)[['baseline', 'TreeTagger', r'\textsc{c}ore\textsc{nlp}']] \
        .rename(to_sc, axis='columns').plot.bar(ax=ax, width=w)
    ax.tick_params(axis='x', rotation=20)
    ax.legend(loc=legend_pos)
    ax.set_title('Utility')
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y')
    ax.set_ylim(**(lambda d: d.update(acc_ylim) or d)(dict(bottom=.4, top=1)))
    start, end = ax.get_ylim()
    ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=False, labelright=False)
    ax.yaxis.set_ticks(np.arange(start, end + acc_ticks_sep, acc_ticks_sep))

    ax = ax3
    df.loc['$|\hat{Y}|$'].unstack(-1)[['baseline', 'TreeTagger', r'\textsc{c}ore\textsc{nlp}']] \
        .rename(to_sc, axis='columns').plot.bar(ax=ax, width=w, legend=False)
    ax.tick_params(axis='x', rotation=20)
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y')
    ax.set_ylim(**(lambda d: d.update(setsize_ylim) or d)(dict(bottom=None, top=None)))
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end + setsize_ticks_sep, setsize_ticks_sep))
    ax.set_title('Set Size')
    f.savefig(outdir + savefile[:-4] + '-all.pdf', bbox_inches='tight')
