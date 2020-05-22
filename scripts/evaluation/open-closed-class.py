import json
import shelve
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from scripts.util.func import param_to_key
from setpos.data.split import load, MCInDocSplitter, is_masked
from setpos.tagger import CoreNLPTagger
from setpos.util import stopwatch, draw_cd_diagram

if __name__ == '__main__':
    SEED, n = 1, 15
    toks, tags, groups = load(
        tag_prefix_masks=[])  # load(tag_prefix_masks=[])  #[l[:3000] for l in load(tag_prefix_masks=[])]    #
    opt = {"augment_setvalued_targets": False, "filter_tags": [],
           "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1",
                                    "-rareWordMinFeatureThresh", "1", "-rareWordThresh", "6", "-sigmaSquared",
                                    "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch",
                                    "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wordTag(0,-1),words(-3,2)"]}
    opt_set_open_classes = opt.copy()
    opt_set_open_classes['corenlp_train_params'] = opt_set_open_classes['corenlp_train_params'] + ['-openClassTags',
                                                                                                   '"ADJA ADJA<VVPP ADJA<VVPS ADJD ADJN ADJN<VVPP ADJS ADJV CARDA CARDN CARDS NA NE VAFIN VAFIN.* VAFIN.ind VAFIN.konj VAINF VAPP VKFIN.* VKFIN.ind VKFIN.konj VKINF VKPP VKPS VMFIN.* VMFIN.ind  VMFIN.konj VMINF VVFIN.* VVFIN.ind VVFIN.konj VVIMP VVINF VVPP VVPS OA XY FM"',
                                                                                                   '-lang', '']
    opt_lang_english = opt.copy()
    opt_lang_english['corenlp_train_params'] = opt_lang_english['corenlp_train_params'] + ['-lang', 'english']
    opt_lang_german = opt.copy()
    opt_lang_german['corenlp_train_params'] = opt_lang_german['corenlp_train_params'] + ['-lang', 'german']
    opt_learn_closed = opt.copy()
    opt_learn_closed['corenlp_train_params'] = opt_learn_closed['corenlp_train_params'] + ['-lang', '',
                                                                                           '-learnClosedClassTags']
    paramspace = [opt,
                  opt_set_open_classes,
                  opt_learn_closed,
                  # opt_lang_english,
                  # opt_lang_german,
                  # dict(augment_setvalued_targets='False',
                  #     corenlp_train_params=['--arch', 'words(-3, 1),lowercasewords(-3, 1),biwords(-3, 2)',
                  #                           'order(2),prefixsuffix(2, 3),'
                  #                           'prefix(1, -1),prefix(2, 0),prefix(3, -2),prefix(3, 0),'
                  #                           'suffix(1, -2),suffix(1, 0),suffix(2, 0),suffix(3, 1)'
                  #                           '--curWordMinFeatureThreshold', '3',
                  #                           '--minFeatureThreshold', '4',
                  #                           '--rareWordMinFeatureThresh', '2',
                  #                           '--rareWordThresh', '48', '--sigmaSquared', '0.6731703824144988',
                  #                           '--veryCommonWordThresh', '324'],
                  #     filter_tags=('XY',))
                  # dict(filter_tags=['$'], augment_setvalued_targets=False, corenlp_train_params=[
                  #    '-curWordMinFeatureThreshold', '1', '-minFeatureThreshold', '1', '-rareWordMinFeatureThresh', '6',
                  #    '-rareWordThresh', '23', '-sigmaSquared', '1.4288954120673443', '-veryCommonWordThresh', '151']),
                  # dict(filter_tags=['FM']),
                  # dict(clf=IntergrammTagger, filter_tags=['FM']),
                  # dict(corenlp_train_params=['-openClassTags',
                  #                           '"ADJA ADJA<VVPP ADJA<VVPS ADJD ADJN ADJN<VVPP ADJS ADJV CARDA CARDN CARDS NA NE VAFIN VAFIN.* VAFIN.ind VAFIN.konj VAINF VAPP VKFIN.* VKFIN.ind VKFIN.konj VKINF VKPP VKPS VMFIN.* VMFIN.ind  VMFIN.konj VMINF VVFIN.* VVFIN.ind VVFIN.konj VVIMP VVINF VVPP VVPS OA"',
                  #                           '-lang', '']),
                  # dict(corenlp_train_params=['-openClassTags',
                  #                           '"ADJA ADJA<VVPP ADJA<VVPS ADJD ADJN ADJN<VVPP ADJS ADJV CARDA CARDN CARDS NA NE VAFIN VAFIN.* VAFIN.ind VAFIN.konj VAINF VAPP VKFIN.* VKFIN.ind VKFIN.konj VKINF VKPP VKPS VMFIN.* VMFIN.ind  VMFIN.konj VMINF VVFIN.* VVFIN.ind VVFIN.konj VVIMP VVINF VVPP VVPS OA"',
                  #                           '-lang', '',
                  #                           '-arch',
                  #                           'words(-2,1),order(-2,1),twoTags(-1,1), wordTag(0,-1),wordTag(0,1),biwords(-1,1)'],
                  #     memlimit='32g'),
                  # dict(filter_tags=['$'])
                  # dict(corenlp_infer_params=['-deterministicTagExpansion']),
                  # dict(tag_expansion_rules=[{'DSG', 'DGN'}]),
                  # dict(tag_expansion_rules=[{'DDSA', 'DDD'}]),
                  # dict(tag_expansion_rules=[{'DSG', 'DGN'}, {'DDSA', 'DDD'}, {'AVG', 'PAVG'}]),
                  # dict(tag_expansion_rules=[{'VKFIN.konj', 'VKFIN.ind', 'VKFIN.*', 'VAFIN.konj', 'VAFIN.ind', 'VAFIN.*'}]),
                  # dict(tag_expansion_rules=[{'DSG', 'DGN'}, {'DDSA', 'DDD'}]),
                  # dict(augment_setvalued_targets=True),
                  # dict(corenlp_train_params=['-arch', 'bidirectional5words']),
                  # dict(corenlp_train_params=['-arch', 'words(-2,1),order(-2,1),twoTags(-1,1), wordTag(0,-1),wordTag(0,1),biwords(-1,1)'],    memlimit='32g'),
                  # dict(corenlp_train_params=['-arch', 'words(-1,1),order(-1,1),twoTags(-1,1), wordTag(0,-1),wordTag(0,1),biwords(-1,1)']),
                  # dict(corenlp_train_params=['-arch', 'bidirectional'], memlimit='16g'),
                  # dict(corenlp_train_params=['-arch', 'naacl2003unknowns']),
                  # dict(corenlp_train_params=['-arch', 'left3words,suffix(1,0),suffix(2,0),suffix(3,0)']),
                  # dict(corenlp_train_params=['-arch', 'left3words,suffix(1,0),suffix(2,0),suffix(3,0),prefix(1,0),prefix(2,0),prefix(3,0)']),
                  ]

    # check if all training parameters for all runs are identical. If yes, we can reuse the fitted classifier
    reuse_training = len(set([json.dumps(d.get('corenlp_train_params', [])) for d in paramspace]))

    # train - test split
    train, test = next(MCInDocSplitter(seed=SEED).split(toks, tags, groups))

    # take the training data for train/eval cross-validation
    toks, tags, groups = [l[train] for l in [toks, tags, groups]]

    with shelve.open(f'result_seed={SEED}_n={len(toks)}_smac-validation2.pkl') as db:
        for params in paramspace:
            param_as_key = param_to_key(params)
            if 'filter_tags' in params:
                mask = np.vectorize(partial(is_masked, prefixes=params['filter_tags']))(tags)
                toks, tags, groups = [a[mask] for a in [toks, tags, groups]]
            clf_class = params.get('clf', CoreNLPTagger)

            # remove higher level parameters
            params = {k: v for k, v in params.items() if k not in {'filter_tags', 'clf'}}

            scores = np.array(db.get(param_as_key, []))
            clf = clf_class(**params)

            with stopwatch():
                print('Params', param_as_key)
                if n > len(scores):
                    try:
                        new_scores = cross_val_score(clf, toks, tags, groups,
                                                     # cv=KFoldInDocSplitter(k=5, seed=0), n_jobs=-1)
                                                     cv=MCInDocSplitter(splits=n, seed=SEED), n_jobs=8)
                    except RuntimeError as e:
                        print(e)
                        print('error')
                        continue
                    scores = np.concatenate((scores, new_scores))
                db[param_as_key] = scores
                scores = scores[:n]
                print(f"total accuracy: {scores.mean():.2%} ± {scores.std():.2%}")

        df = pd.DataFrame(
            [dict(classifier_name=k, accuracy=v, dataset_name=i) for k, vs in db.items() for i, v in
             enumerate(vs[:n]) if k in set(map(param_to_key, paramspace))])
        draw_cd_diagram(df, verbose=True)

'''
Params {"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1", "-rareWordMinFeatureThresh", "1", "-rare
WordThresh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0
),wordTag(0,-1),words(-3,2)"]}                                                                                                                                                                   
total accuracy: 86.05% _ 0.41%                                                                                                                                                                   
elapsed time: 0.0s                                                                                                                                                                               
Params {"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1", "-rareWordMinFeatureThresh", "1", "-rare
WordThresh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0
),wordTag(0,-1),words(-3,2)", "-openClassTags", "\"ADJA ADJA<VVPP ADJA<VVPS ADJD ADJN ADJN<VVPP ADJS ADJV CARDA CARDN CARDS NA NE VAFIN VAFIN.* VAFIN.ind VAFIN.konj VAINF VAPP VKFIN.* VKFIN.ind
 VKFIN.konj VKINF VKPP VKPS VMFIN.* VMFIN.ind  VMFIN.konj VMINF VVFIN.* VVFIN.ind VVFIN.konj VVIMP VVINF VVPP VVPS OA\"", "-lang", ""]}                                                          
total accuracy: 85.51% _ 0.46%                                                                                                                                                                   
elapsed time: 959.2s                                                                                                                                                                             
Params {"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1", "-rareWordMinFeatureThresh", "1", "-rare
WordThresh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0
),wordTag(0,-1),words(-3,2)", "-lang", "english"]}                                                                                                                                               
total accuracy: 86.67% _ 0.47%                                                                                                                                                                   
elapsed time: 1044.2s                                                                                                                                                                            
p-values:                                                                                                                                                                                        
('{"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1", "-rareWordMinFeatureThresh", "1", "-rareWordT
hresh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wor
dTag(0,-1),words(-3,2)", "-lang", "english"]}', '{"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1"
, "-rareWordMinFeatureThresh", "1", "-rareWordThresh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suff
ix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wordTag(0,-1),words(-3,2)", "-openClassTags", "\\"ADJA ADJA<VVPP ADJA<VVPS ADJD ADJN ADJN<VVPP ADJS ADJV CARDA CARDN CARDS NA NE VAFIN VAFIN.* VAFIN.
ind VAFIN.konj VAINF VAPP VKFIN.* VKFIN.ind VKFIN.konj VKINF VKPP VKPS VMFIN.* VMFIN.ind  VMFIN.konj VMINF VVFIN.* VVFIN.ind VVFIN.konj VVIMP VVINF VVPP VVPS OA\\"", "-lang", ""]}', 0.000654958
3433856954, True) 
('{"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1", "-rareWordMinFeatureThresh", "1", "-rareWordT
hresh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wor
dTag(0,-1),words(-3,2)", "-lang", "english"]}', '{"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1"
, "-rareWordMinFeatureThresh", "1", "-rareWordThresh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suff
ix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wordTag(0,-1),words(-3,2)"]}', 0.0012063162374292998, True)

('{"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1", "-rareWordMinFeatureThresh", "1", "-rareWordT
hresh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wor
dTag(0,-1),words(-3,2)", "-openClassTags", "\\"ADJA ADJA<VVPP ADJA<VVPS ADJD ADJN ADJN<VVPP ADJS ADJV CARDA CARDN CARDS NA NE VAFIN VAFIN.* VAFIN.ind VAFIN.konj VAINF VAPP VKFIN.* VKFIN.ind VKF
IN.konj VKINF VKPP VKPS VMFIN.* VMFIN.ind  VMFIN.konj VMINF VVFIN.* VVFIN.ind VVFIN.konj VVIMP VVINF VVPP VVPS OA\\"", "-lang", ""]}', '{"augment_setvalued_targets": false, "filter_tags": [], "
corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1", "-rareWordMinFeatureThresh", "1", "-rareWordThresh", "6", "-sigmaSquared", "0.7676194187745077", "-veryC
ommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wordTag(0,-1),words(-3,2)"]}', 0.010593539088689496, True)

Wins
{"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1", "-rareWordMinFeatureThresh", "1", "-rareWordThr
esh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wordT
ag(0,-1),words(-3,2)", "-lang", "english"]}     
                                                                                                                                    14.0
{"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1", "-rareWordMinFeatureThresh", "1", "-rareWordThr
esh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wordT
ag(0,-1),words(-3,2)", "-openClassTags", "\"ADJA ADJA<VVPP ADJA<VVPS ADJD ADJN ADJN<VVPP ADJS ADJV CARDA CARDN CARDS NA NE VAFIN VAFIN.* VAFIN.ind VAFIN.konj VAINF VAPP VKFIN.* VKFIN.ind VKFIN.
konj VKINF VKPP VKPS VMFIN.* VMFIN.ind  VMFIN.konj VMINF VVFIN.* VVFIN.ind VVFIN.konj VVIMP VVINF VVPP VVPS OA\"", "-lang", ""]}     0.0
{"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1", "-rareWordMinFeatureThresh", "1", "-rareWordThr
esh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wordT
ag(0,-1),words(-3,2)"]}        

Avarage Ranks
{"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1", "-rareWordMinFeatureThresh", "1", "-rareWordThr
esh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wordT
ag(0,-1),words(-3,2)", "-openClassTags", "\"ADJA ADJA<VVPP ADJA<VVPS ADJD ADJN ADJN<VVPP ADJS ADJV CARDA CARDN CARDS NA NE VAFIN VAFIN.* VAFIN.ind VAFIN.konj VAINF VAPP VKFIN.* VKFIN.ind VKFIN.
konj VKINF VKPP VKPS VMFIN.* VMFIN.ind  VMFIN.konj VMINF VVFIN.* VVFIN.ind VVFIN.konj VVIMP VVINF VVPP VVPS OA\"", "-lang", ""]}    2.933333
{"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1", "-rareWordMinFeatureThresh", "1", "-rareWordThr
esh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wordT
ag(0,-1),words(-3,2)"]}                         
                                                                                                                                    2.000000
{"augment_setvalued_targets": false, "filter_tags": [], "corenlp_train_params": ["-curWordMinFeatureThreshold", "4", "-minFeatureThreshold", "1", "-rareWordMinFeatureThresh", "1", "-rareWordThr
esh", "6", "-sigmaSquared", "0.7676194187745077", "-veryCommonWordThresh", "234", "-arch", "order(-2,0),prefix(1,0),prefix(2,0),prefix(3,0),suffix(2,0),suffix(3,0),suffix(4,0),suffix(5,0),wordT
ag(0,-1),words(-3,2)", "-lang", "english"]}     
                                                                                                                                    1.066667

'''
