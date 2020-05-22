import shelve
from functools import cmp_to_key
from itertools import groupby, chain, combinations
from typing import Iterable, List

import pandas as pd
from sklearn.model_selection import cross_val_score

from setpos.data.split import MCInDocSplitter, load
from setpos.tagger import CoreNLPTagger
from setpos.util import stopwatch


def fit_and_score(clf, Xtrain, ytrain, Xeval, yeval):
    clf.fit(Xtrain, ytrain)
    return clf.score(Xeval, yeval, long_result=True)


def evaluate(rules, n=1):
    with stopwatch(False) as sw:
        clf = CoreNLPTagger(tag_expansion_rules=rules)

        scores = pd.Series(
            cross_val_score(clf, toks, tags, groups, cv=MCInDocSplitter(train_frac=.8, splits=n), n_jobs=-1))
        acc, ci = scores.mean(), scores.sem() * 1.96

    # print accuracy and current ruleset
    print(f'{acc:.2%}±{ci:.2%} with {curr_rules} took {sw.interval:.2f}s')
    return dict(acc=acc, ci=ci, n=n)


TAGS = ['ADJA', 'ADJA<VVPP', 'ADJA<VVPS', 'ADJD', 'ADJN', 'ADJN<VVPP', 'ADJS', 'ADJV', 'APPO', 'APPR', 'AVD', 'AVG',
        'AVNEG', 'AVREL', 'AVW', 'CARDA', 'CARDN', 'CARDS', 'DDA', 'DDART', 'DDD', 'DDN', 'DDS', 'DDSA', 'DGA', 'DGN',
        'DGS', 'DIA', 'DIART', 'DID', 'DIN', 'DIS', 'DNEGA', 'DNEGS', 'DPDS', 'DPOSA', 'DPOSD', 'DPOSGEN', 'DPOSN',
        'DPOSS', 'DRELA', 'DRELS', 'DWS', 'FM', 'KO*', 'KOKOM', 'KON', 'KOUS', 'NA', 'NE', 'OA', 'PAVAP', 'PAVD',
        'PAVG', 'PAVREL', 'PG', 'PI', 'PKOR', 'PNEG', 'PPER', 'PRF', 'PTKA', 'PTKANT', 'PTKG', 'PTKN', 'PTKNEG',
        'PTKREL', 'PTKVZ', 'PTKZU', 'VAFIN', 'VAFIN.*', 'VAFIN.ind', 'VAFIN.konj', 'VAINF', 'VAPP', 'VKFIN.*',
        'VKFIN.ind', 'VKFIN.konj', 'VKINF', 'VKPP', 'VKPS', 'VMFIN.*', 'VMFIN.ind', 'VMFIN.konj', 'VMINF', 'VVFIN.*',
        'VVFIN.ind', 'VVFIN.konj', 'VVIMP', 'VVINF', 'VVPP', 'VVPS']


def lazy_setdefault_in_shelf(d: dict, shelfkey, key, value: callable):
    internald = d[shelfkey]
    if key not in internald:
        v = value()
        internald.setdefault(key, v)
        d[shelfkey] = internald
    else:
        r = internald[key]
        if not isinstance(r, dict):
            if isinstance(r, tuple):
                internald[key] = dict(acc=r[0], n=6)
            elif isinstance(r, float):
                internald[key] = dict(acc=r, n=6)
            d[shelfkey] = internald

    r = internald[key]
    # convert old values
    return r['acc']


def powerset(iterable, min_len=2, max_len=10):
    "powerset([1,2,3]) --> (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(min_len, min(max_len, len(s)) + 1))


def gen_base_rules(max_len_powerset=1) -> Iterable[set]:
    def cmp(t1, t2):
        t1, t2 = [set([t] + t.split("<") + [t.strip('.konj')]) for t in [t1, t2]]
        return 0 if len(t1 & t2) > 0 else -1

    equiv_sets = set([frozenset(g_) for _, g_ in groupby(sorted(TAGS), key=cmp_to_key(cmp))]) \
                 | set([frozenset([tag for tag in TAGS if tag.startswith(p)]) for p in
                        ['ADJ', 'ADJD', 'ADJN', 'ADJS', 'APP', 'AV', 'CARD',
                         'DD', 'DG', 'DI', 'DNEG', 'DD', 'DPOS', 'DREL',
                         'KO', 'N', 'PAV', 'PTK', 'VA', 'VK', 'VV']]) \
                 | set([frozenset([tag for tag in TAGS if tag.split('.')[0].endswith(suff)]) for suff in
                        ['AVD', 'AVG', 'AVREL', 'NEG', 'FIN', 'IMP', 'INF', 'PP', 'PS']])

    # generate all powersets of rules
    equiv_sets = set(chain.from_iterable((powerset(r) for r in equiv_sets)))

    # sort all equiv-sets by their size, such that we first apply small rules if supersets of rules exist
    p_set = list(set((powerset(TAGS, max_len=max_len_powerset))) - equiv_sets)
    # since the sorting is stable, the handcrafted ones will be searched first
    return sorted(filter(lambda s: len(s) > 1, list(equiv_sets) + p_set), key=len)


def merge(rules: List[frozenset]) -> Iterable[frozenset]:
    merged = rules.copy()
    for rule in rules:
        if any((rule < r for r in merged)):
            merged.remove(rule)
    return merged


def hand_crafted_rules():
    return [{'DSG', 'DGN'}, {'DDSA', 'DDD'}]


if __name__ == '__main__':
    with shelve.open('results_handcraft.pkl') as db, stopwatch():
        try:
            # generate all base-rules
            # rules = [r for r in gen_base_rules(max_len_powerset=1)]
            rules = [r for r in hand_crafted_rules()]
            print(f'generated {len(rules)} rules', flush=True)

            toks, tags, groups = load()  # [l[:3000] for l in load()]  #
            curr_rules = []
            if 'rejected_rules' not in db:
                db['rejected_rules'] = []
            if 'results' not in db:
                db['results'] = {}
            else:
                print(f'loading {len(db["results"])} previous results')

            best_acc = acc = initial_acc = lazy_setdefault_in_shelf(db, 'results', frozenset(), lambda: evaluate(()))

            for rule in rules:
                rule = frozenset(rule)
                with stopwatch(False) as sw:
                    if any((r.issubset(rule) for r in db['rejected_rules'])):
                        # a subset of this rule aready reduced accuracy
                        # since we generate all powersets of rules, all other subsets will be checked anyways
                        # therefore its not credible that it will lead to an improvement, we wont get otherwise
                        continue
                    if sw.interval > 2:
                        print('subset_rule evaluation took: {sw.interval}s')

                # add rule to active set
                print('investigating', rule)
                curr_rules.append(rule)
                merged_rules = merge(curr_rules)

                # evaluate active set
                acc = lazy_setdefault_in_shelf(db, 'results', frozenset(merged_rules), lambda: evaluate(merged_rules))

                # if accuracy dropped, remove last rule
                if best_acc > acc:
                    db['rejected_rules'] = db['rejected_rules'] + [curr_rules.pop()]
                else:
                    best_acc = acc
                    print(f'total acc gain: {acc - initial_acc:.2%}')
        finally:
            pd.DataFrame().from_dict(db['results'], orient='index').to_csv('results.csv')
