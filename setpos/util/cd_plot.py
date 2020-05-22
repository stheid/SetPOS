# Author: Hassan Ismail Fawaz <hassan.ismail-fawaz@uha.fr>
#         Germain Forestier <germain.forestier@uha.fr>
#         Jonathan Weber <jonathan.weber@uha.fr>
#         Lhassane Idoumghar <lhassane.idoumghar@uha.fr>
#         Pierre-Alain Muller <pierre-alain.muller@uha.fr>
# License: GPL3
import operator
from contextlib import redirect_stdout, redirect_stderr
from sys import stdout

import numpy as np
import pandas as pd
# from Orange import evaluation
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon


def avg_ranks(df_perf, alpha=.05):
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['classifier_name'])

    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)]. \
        sort_values(['classifier_name', 'dataset_name'])

    # get the rank data
    rank_data = np.array(sorted_df_perf['accuracy']).reshape(len(classifiers), max_nb_datasets)

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=
    np.unique(sorted_df_perf['dataset_name']))

    # number of wins
    print('Wins')
    dfff = df_ranks.rank(ascending=False)
    print(dfff[dfff == 1.0].sum(axis=1))
    print()

    # average the ranks
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)

    print('Avarage Ranks')
    print(average_ranks)
    print()
    # return the average ranks
    return average_ranks


def draw_cd_diagram(df_perf, alpha=0.05, verbose=False):
    """
    Draws the critical difference diagram given the list of pairwise classifiers that are
    significant or not
    """
    with redirect_stdout(None if not verbose else stdout), redirect_stderr(None):
        wilcoxon_holm(df_perf, alpha=alpha)
        ranks = avg_ranks(df_perf, alpha=alpha)


#      cd = evaluation.compute_CD(ranks.values, df_perf.groupby('classifier_name').size().min())
#     evaluation.graph_ranks(ranks.values, ranks.keys(), cd=cd, width=20, reverse=True, textspace=8,
#                           filename='cd-diagram.pdf')


def wilcoxon_holm(df_perf, alpha=0.05):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """
    # print(pd.unique(df_perf['classifier_name']))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['classifier_name'])
    # test the null hypothesis using friedman before doing a post-hoc analysis
    try:
        friedman_p_value = friedmanchisquare(*(
            np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
            for c in classifiers))[1]
        if friedman_p_value >= alpha:
            raise RuntimeError('the null hypothesis over the entire classifiers cannot be rejected')
    except ValueError:
        pass
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_1]['accuracy']
                          , dtype=np.float64)
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_2]
                              ['accuracy'], dtype=np.float64)
            # calculate the p_value
            try:
                p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            except ValueError:
                p_value = 1
            # appens to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            # stop
            break

    print('p-values:')
    for p in p_values:
        print(p)
        print()


if __name__ == '__main__':
    df_perf = pd.read_csv('example.csv', index_col=False)

    draw_cd_diagram(df_perf=df_perf, verbose=True)
