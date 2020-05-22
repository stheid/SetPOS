import re

import numpy as np

from setpos.data.evaluate.io_ import load

df, tagsdict = load()


def avg_jaccard_sim_of_top_k_vs_allowed():
    sim = np.empty(len(df))
    for i, row in df.iterrows():
        probas = np.array(eval(row.labelposterior))
        valid_tag_indices = set([tagsdict.inv[t] for t in re.split(", ", re.match("\[(.*)\]", row.constrainedtags)[1])])
        # partition will try sort ascending, but we need to have the top-k, therefore the negation and taking the first k elements
        top_l_tag_indices = set(np.argpartition(probas * -1, len(valid_tag_indices) - 1)[:len(valid_tag_indices)])

        # average jaccard similarity
        sim[i] = len(valid_tag_indices & top_l_tag_indices) / len(valid_tag_indices | top_l_tag_indices)

    return sim.mean()


def avg_proba_explained_by_valid():
    probs = np.empty(len(df))
    for i, row in df.iterrows():
        probas = np.array(eval(row.labelposterior))
        valid_tag_indices = np.array(
            [tagsdict.inv[t] for t in re.split(", ", re.match("\[(.*)\]", row.constrainedtags)[1])])

        probs[i] = np.exp(probas[valid_tag_indices]).sum()

    return probs.mean()


if __name__ == '__main__':
    print(avg_jaccard_sim_of_top_k_vs_allowed())
    print(avg_proba_explained_by_valid())
