import json
from itertools import takewhile

import pandas as pd
from sklearn.metrics import confusion_matrix

from setpos.data.split import load
from setpos.util import score, create_g_alpha_beta, create_ubop_predictor, util


def print_stats(data: pd.DataFrame, label: str, fun: callable, fmt=".2%"):
    """
    generates something like this:

    >           total   known   unk
    > avg util: 0.80   (0.83 ; 0.28 )

    :param data:
    :param label:
    :param fun: converts dataframe into a metric that can be formated as a string
    :param fmt:
    :return:
    """
    unkmask = data["isunk"]

    # precompute total for formating
    unk, known, total = fun(data[unkmask]), fun(data[~unkmask]), fun(data)
    totalstr = f"{total:^5{fmt}}"

    # calculate format sizes
    formatsize = max(5, len(totalstr))
    leftskip = len(f"{label}:")
    print(f"{'':{leftskip}} {'total':^{formatsize}}   {'known':^{formatsize}}  {'unk':^{formatsize}}")
    print(
        f"{label}: {totalstr}  ({known:^{formatsize}{fmt}}; {unk:^{formatsize}{fmt}})")
    print()

    return {label: {"unk": unk, "known": known, "total": total}}


def print_hdr(head):
    print(f"\n{head.title()}\n" + '⎺' * len(head))


def print_classical_pred_stats(data, tagsdict, plot_confustion_matrix=False):
    print_hdr("classical")
    result = dict()
    data.target = data.target.apply(json.loads)

    data = data.assign(iscorrect=data.apply(lambda x: score(x.target, x.pred), axis=1),
                       # (#40) upper bound of CoreNLP, words that it MUST misclassify as the target is not in possible tags
                       forced_error=data.apply(lambda x: len(x.target.keys() & x.constrainedtags) == 0, axis=1))
    print(f"words unknown: {data.isunk.mean():.2%}")
    result.update(print_stats(data, "accuracy", lambda x: x.iscorrect.mean()))
    print(f"avg sent correct: {data.groupby(['sentID']).iscorrect.all().mean():.2%}\n")
    result.update(print_stats(data, "upper bound accuracy", lambda x: 1 - x.forced_error.mean()))
    print(f"avg sent correct (upper bound): {1 - data.groupby(['sentID']).forced_error.any().mean():.2%}\n")

    if plot_confustion_matrix:
        # unfortunately, not all tags that are used in the test-data are known during training, therefore only a part of the testdata,
        # can be used to make a proper confusion matrix (the mainly for retrieving the correct labels for each tag)
        # but actually even without filtering, the resulting plot is very uninformative
        data = data.assign(both_tags_in_tagset=data.apply(
            lambda x: x.target in tagsdict.values() and x.pred in tagsdict.values(), axis=1))
        tags = set(data[data.both_tags_in_tagset].target) | set(data[data.both_tags_in_tagset].pred)
        sorted_escaped_tags = [s.replace('$', r'\$') for s in sorted(tags, key=tagsdict.inv.__getitem__)]
        conf_mat = confusion_matrix(data[data.both_tags_in_tagset].target.apply(tagsdict.inv.__getitem__),
                                    data[data.both_tags_in_tagset].pred.apply(tagsdict.inv.__getitem__))

        for tag, preds in sorted([(sorted_escaped_tags[i], preds) for i, preds in enumerate(conf_mat)]):
            # for each ground truth tag we print the predictions sorted by their occurrence
            if sum(preds) < 10:
                # skip vary uncommon tags (the sum of all predictions is the total occurrence als ground truth)
                continue
            preds = takewhile(lambda n_errors: n_errors[1] > 2,
                              sorted(enumerate(preds), key=lambda x: x[1], reverse=True))
            print(tag, "::", ", ".join([f"{sorted_escaped_tags[j]}({n})" for j, n in preds]))

        # sns.heatmap(conf_mat, xticklabels=sorted_escaped_tags, yticklabels=sorted_escaped_tags)
        # plt.show()

    return data, pd.DataFrame.from_dict(result, "index")


def print_set_valued_pred_stats(data, tagsdict, show_unconstrained=False, g=None):
    result = dict()

    UNK = True
    g_pair = {not UNK: g or create_g_alpha_beta(1, 1, 92), UNK: g or create_g_alpha_beta(1, 1, 92)}
    set_predictor = {isunk: create_ubop_predictor(g_) for isunk, g_ in g_pair.items()}

    # unconst_set_pred
    labels = tuple([v for i, v in sorted(tagsdict.items())])
    for is_constrained in ([True, False] if show_unconstrained else [True]):
        prefix = "" if is_constrained else "un"

        if is_constrained:
            # pick posterior probabilities by the indices of the constrained tags using array indexing
            data = data.assign(constrained_posterior=data.apply(
                lambda x: x.posterior[[tagsdict.inv[t] for t in x.constrainedtags]], axis=1))
            data = data.assign(const_set_pred=data.apply(
                lambda x: set_predictor[x.isunk](x['constrained_posterior'], x.constrainedtags, True), axis=1))
        else:
            data = data.assign(**{prefix + "const_set_pred": data.apply(
                lambda x: set_predictor[x.isunk](x['posterior'], labels, True), axis=1)})
        data = data.assign(**{prefix + "const_set_size": data[prefix + "const_set_pred"].apply(len)})
        data = data.assign(**{prefix + "const_util": data.apply(
            lambda x: util(g_pair[x.isunk], x['target'], x[prefix + 'const_set_pred']), axis=1)})
        data = data.assign(**{prefix + "const_recall": data.apply(
            lambda x: util(lambda _: 1, x['target'], x[prefix + 'const_set_pred']), axis=1)})
        data = data.assign(**{prefix + "const_precision": data.apply(
            lambda x: util(lambda s: 1 / s, x['target'], x[prefix + 'const_set_pred']), axis=1)})
        data = data.assign(**{prefix + "original_pred_in_set_pred":
                                  data.apply(lambda x: x.pred in x[prefix + 'const_set_pred'], axis=1)})

        print_hdr(prefix + "constrained")
        result.update(print_stats(data, "avg setsize", lambda x: x[prefix + "const_set_size"].mean(), ".2f"))
        result.update(print_stats(data, "avg util", lambda x: x[prefix + "const_util"].mean()))
        result.update(print_stats(data, "avg recall", lambda x: x[prefix + "const_recall"].mean()))
        print(f"avg sent recall {data.groupby(['sentID'])[[prefix + 'const_recall']].all().mean().iloc[0]:.2%}")
        result.update(print_stats(data, "avg precision", lambda x: x[prefix + "const_precision"].mean()))
        result.update(print_stats(data, "avg agreement of set-prediction with default prediction",
                                  lambda x: x[prefix + "original_pred_in_set_pred"].mean()))

        return data, pd.DataFrame.from_dict(result, 'index')


if __name__ == '__main__':
    print(score(dict(a=1, b=1), {'b'}))
    print(score(dict(a=.6, b=.3), {'b'}))
    print(score(dict(a=1), {'b'}))
    print(score(dict(b=1), {'b'}))

    df, tagsdict = load()
    df, _ = print_classical_pred_stats(df, tagsdict)
    df, _ = print_set_valued_pred_stats(df, tagsdict, True)
    df.to_excel("out.xlsx")
