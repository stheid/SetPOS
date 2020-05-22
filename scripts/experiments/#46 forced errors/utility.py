from functools import cmp_to_key
from itertools import groupby

import pandas as pd

from setpos.tagger import create_g_alpha_beta
from setpos.util import stopwatch

TAGS = ['ADJA', 'ADJA<VVPP', 'ADJA<VVPS', 'ADJD', 'ADJN', 'ADJN<VVPP', 'ADJS', 'ADJV', 'APPO', 'APPR', 'AVD',
        'AVG', 'AVNEG', 'AVREL', 'AVW', 'CARDA', 'CARDN', 'CARDS', 'DDA', 'DDART', 'DDD', 'DDN', 'DDS', 'DDSA',
        'DGA', 'DGN', 'DGS', 'DIA', 'DIART', 'DID', 'DIN', 'DIS', 'DNEGA', 'DNEGS', 'DPDS', 'DPOSA', 'DPOSD',
        'DPOSGEN', 'DPOSN', 'DPOSS', 'DRELA', 'DRELS', 'DWS', 'FM', 'KO*', 'KOKOM', 'KON', 'KOUS', 'NA', 'NE',
        'OA', 'PAVAP', 'PAVD', 'PAVG', 'PAVREL', 'PG', 'PI', 'PKOR', 'PNEG', 'PPER', 'PRF', 'PTKA', 'PTKANT',
        'PTKG', 'PTKN', 'PTKNEG', 'PTKREL', 'PTKVZ', 'PTKZU', 'VAFIN', 'VAFIN.konj', 'VAINF', 'VAPP', 'VKFIN',
        'VKFIN.konj', 'VKINF', 'VKPP', 'VKPS', 'VMFIN', 'VMFIN.konj', 'VMINF', 'VVFIN', 'VVFIN.konj', 'VVIMP',
        'VVINF', 'VVPP', 'VVPS']


def gen_base_rules():
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

    # sort all equiv-sets by their size, such that we first apply small rules if supersets of rules exist
    return sorted(filter(lambda s: len(s) > 1, equiv_sets), key=len)


def util(g, target, pred_set):
    return g(len(pred_set)) if target in pred_set else 0


def get_tag_expander():
    equiv_sets = [{'DRELA', 'DRELS'}, {'DNEGS', 'DNEGA'}, {'VAFIN', 'VAFIN.konj'}, {'APPR', 'APPO'}, {'AVD', 'PAVD'},
                  {'PTKNEG', 'PNEG', 'AVNEG'}, {'ADJA<VVPS', 'VVPS', 'VKPS'}]  # gen_base_rules()

    def expand_tags(tagset):
        tagset = set(tagset)

        added_tags = set()

        # merge tags into groups by prefix. if any group representant is avail, we add the whole group

        for set_ in equiv_sets:
            if tagset & set_:
                added_tags.update(set_)
        return tagset | added_tags

    return expand_tags


expand_tags = get_tag_expander()

if __name__ == '__main__':
    with stopwatch():
        df = pd.read_excel('2019-08-27 18:01:12: constraint_set-no_expansion.xlsx', keep_default_na=False)
        df.constrainedtags = df.constrainedtags.apply(eval)

        # tuned to give a relatively high utility for the default case
        g = create_g_alpha_beta(1, 1.5, 87)
        df = df.assign(util=df.apply(lambda x: util(g, x.target, x.constrainedtags), axis=1))
        df = df.assign(extended_tags=df.apply(lambda x: expand_tags(x.constrainedtags), axis=1))
        df = df.assign(extended_util=df.apply(lambda x: util(g, x.target, x.extended_tags), axis=1))
        df = df.assign(extended_forced_err=df.apply(lambda x: x.target not in x.extended_tags, axis=1))

        # apply rules and recalc utility
        print(f'mean utility:\t {df.util.mean():.3f}')
        print(f'forced error rate:\t {df.forced_error.mean():.2%}')
        print()

        print(f'extended mean utility:\t {df.extended_util.mean():.3f}')
        print(f'extended forced error rate:\t {df.extended_forced_err.mean():.2%}')
